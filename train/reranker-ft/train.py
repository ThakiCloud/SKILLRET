"""
Fine-tune Qwen3-Reranker on SkillRet data (multi-GPU DDP).

Supports two loss modes (config: loss_type):

  pointwise (default):
    BCE on P("yes") per (query, document) pair — identical to Qwen3-Reranker
    inference scoring.

  listwise:
    Softmax cross-entropy over the full candidate group per query.
    The ranking score is the log-odds log P("yes")/P("no").  Each query's
    candidates are kept together in the same batch via GroupBatchSampler so
    the softmax denominator is well-defined.  Follows the finding in the
    SkillRouter paper that listwise loss is decisive for homogeneous pools.

Supports hard-negative mining from first-stage retrieval results and
configurable number of negatives per positive pair.

Training data is loaded from the HuggingFace dataset:
    ThakiCloud/SKILLRET

Usage:
    cd /path/to/skillret-benchmark
    source .venv/bin/activate

    # 1) Mine hard negatives (single GPU):
    python train/reranker-ft/train.py --mine-hard-negatives \
        --config train/reranker-ft/configs/qwen3-reranker-0.6b-sft.yaml

    # 2) Train (multi-GPU DDP):
    torchrun --nproc_per_node=8 train/reranker-ft/train.py \
        --config train/reranker-ft/configs/qwen3-reranker-0.6b-sft.yaml
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
os.environ["WANDB_PROJECT"] = "skillret"

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "skillret.config",
    str(Path(__file__).resolve().parent.parent.parent / "skillret" / "config.py"),
)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
SKILL_RERANK_INSTRUCTION = _cfg.SKILL_RERANK_INSTRUCTION
HF_DATASET_ID = _cfg.HF_DATASET_ID

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_hf_dataset(subset: str, split: str = "test") -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load(HF_DATASET_ID, subset, split=split)
    return [dict(row) for row in ds]


def load_train_queries() -> list[dict]:
    return _load_hf_dataset("queries", split="train")


def load_train_skills() -> list[dict]:
    skills = _load_hf_dataset("skills", split="train")
    for s in skills:
        if "skill_md" not in s or not (s.get("skill_md") or "").strip():
            s["skill_md"] = s.get("description", "")
    return skills


def load_eval_queries() -> list[dict]:
    return _load_hf_dataset("queries", split="test")


def load_eval_skills() -> list[dict]:
    skills = _load_hf_dataset("skills", split="test")
    for s in skills:
        if "skill_md" not in s or not (s.get("skill_md") or "").strip():
            s["skill_md"] = s.get("description", "")
    return skills


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    defaults = {
        "max_seq_length": 8192,
        "batch_size": 16,
        "grad_accum": 1,
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "epochs": 1,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 10,
        "bf16": True,
        "seed": 42,
        "wandb_run": None,
        "num_negatives": 7,
        "hard_negatives_file": None,
        "hard_negatives_files": None,
        "hard_neg_skip_top": 20,
        "hard_neg_top_k": 60,
        "per_positive_negatives": False,
        "merge_queries": True,
        "load_best_model_at_end": True,
        "embedding_model": None,
        "eval_first_stage_file": "results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json",
        "eval_max_queries": 500,
        "eval_batch_size": 16,
        "loss_type": "pointwise",
        "fixed_group_size": None,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.001,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    if not cfg.get("model"):
        raise ValueError("Config must specify 'model' (HuggingFace model ID)")
    if not cfg.get("output_dir"):
        raise ValueError("Config must specify 'output_dir'")

    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        cfg["output_dir"] = str(SCRIPT_DIR / out)

    hn = cfg.get("hard_negatives_file")
    if hn and not Path(hn).is_absolute():
        cfg["hard_negatives_file"] = str(PROJECT_ROOT / hn)

    hn_files = cfg.get("hard_negatives_files")
    if hn_files:
        cfg["hard_negatives_files"] = [
            str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p
            for p in hn_files
        ]

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════

def build_skill_text(skill: dict) -> str:
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    body = (skill.get("skill_md") or "").strip()
    return f"{name} | {desc} | {body}"


def build_skill_lookup(skills: list[dict]) -> dict[str, str]:
    return {s["id"]: build_skill_text(s) for s in skills}



def merge_queries(queries: list[dict]) -> list[dict]:
    """Merge rows with the same query ID, unioning their skill_ids."""
    by_id: dict[str, dict] = {}
    for q in queries:
        qid = q["id"]
        if qid in by_id:
            existing = set(by_id[qid]["skill_ids"])
            existing.update(q["skill_ids"])
            by_id[qid]["skill_ids"] = list(existing)
        else:
            by_id[qid] = dict(q)
    merged = list(by_id.values())
    if len(merged) < len(queries):
        logger.info(
            f"  Merged {len(queries):,} rows → {len(merged):,} unique queries "
            f"({len(queries) - len(merged):,} duplicates resolved)"
        )
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Hard-negative mining
# ═══════════════════════════════════════════════════════════════════════════

def mine_hard_negatives(
    embedding_model: str,
    keep_top: int = 100,
    output_file: Path | None = None,
    encode_batch_size: int = 32,
) -> dict[str, list[str]]:
    """Mine hard negatives using a first-stage embedding model.

    Stores a ranked list of the top ``keep_top`` non-GT candidates per query
    (ordered by descending similarity).  The file can then be sliced at
    training time via ``hard_neg_skip_top`` / ``hard_neg_top_k`` config
    options — no need to re-mine for different skip/top_k settings.

    Ground-truth positives are excluded at every stage.
    Automatically uses multi-GPU encoding when multiple GPUs are available.
    """
    import numpy as np

    queries = merge_queries(load_train_queries())
    skills = load_train_skills()
    skill_lookup = {s["id"]: build_skill_text(s) for s in skills}
    skill_ids = list(skill_lookup.keys())
    skill_texts = [skill_lookup[sid] for sid in skill_ids]

    logger.info(f"Mining hard negatives with {embedding_model}")
    logger.info(
        f"  {len(queries):,} queries, {len(skills):,} skills, "
        f"keeping top {keep_top} non-GT candidates per query (ranked)"
    )

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(embedding_model, trust_remote_code=True)

    n_gpu = torch.cuda.device_count()
    use_multi = n_gpu > 1

    if use_multi:
        logger.info(f"Using multi-GPU encoding ({n_gpu} GPUs)...")
        pool = st_model.start_multi_process_pool()

        logger.info("Encoding skills...")
        skill_embs = st_model.encode_multi_process(
            skill_texts, pool, batch_size=encode_batch_size, normalize_embeddings=True,
        )
        from skillret.config import SKILL_QUERY_PROMPT
        query_texts = [SKILL_QUERY_PROMPT + q["query"] for q in queries]
        logger.info("Encoding queries...")
        query_embs = st_model.encode_multi_process(
            query_texts, pool, batch_size=encode_batch_size, normalize_embeddings=True,
        )
        st_model.stop_multi_process_pool(pool)
    else:
        logger.info("Using single-GPU encoding...")
        from skillret.config import SKILL_QUERY_PROMPT

        logger.info("Encoding skills...")
        skill_embs = st_model.encode(
            skill_texts, batch_size=encode_batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        query_texts = [q["query"] for q in queries]
        logger.info("Encoding queries...")
        query_embs = st_model.encode(
            query_texts, batch_size=encode_batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
            prompt=SKILL_QUERY_PROMPT,
        )

    logger.info("Computing similarities and ranking (batched)...")
    hard_negs: dict[str, list[str]] = {}
    skill_embs_t = torch.from_numpy(np.ascontiguousarray(skill_embs)).cuda()

    max_gt = max(len(q["skill_ids"]) for q in queries)
    fetch_k = min(keep_top + max_gt + 10, len(skill_ids))

    batch_sz = 1024
    for start in range(0, len(queries), batch_sz):
        end = min(start + batch_sz, len(queries))
        q_batch = torch.from_numpy(
            np.ascontiguousarray(query_embs[start:end])
        ).cuda()
        scores = q_batch @ skill_embs_t.T
        _, topk_indices = scores.topk(fetch_k, dim=1)
        topk_indices = topk_indices.cpu().tolist()

        for j, q in enumerate(queries[start:end]):
            gt_ids = set(q["skill_ids"])
            neg_ids = []
            for idx in topk_indices[j]:
                sid = skill_ids[idx]
                if sid not in gt_ids:
                    neg_ids.append(sid)
                    if len(neg_ids) >= keep_top:
                        break
            hard_negs[q["id"]] = neg_ids

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(hard_negs, f)
        logger.info(f"Saved hard negatives to {output_file}")
        avg_negs = np.mean([len(v) for v in hard_negs.values()])
        logger.info(f"  Avg negatives per query: {avg_negs:.1f} (top {keep_top})")

    return hard_negs


# ═══════════════════════════════════════════════════════════════════════════
# Dataset construction
# ═══════════════════════════════════════════════════════════════════════════

def build_train_dataset(
    queries: list[dict],
    skill_lookup: dict[str, str],
    num_negatives: int = 7,
    hard_negatives: dict[str, list[str]] | None = None,
    hard_neg_skip_top: int = 0,
    hard_neg_top_k: int | None = None,
    seed: int = 42,
    per_positive_negatives: bool = False,
    do_merge_queries: bool = True,
    fixed_group_size: int | None = None,
    random_neg_sampling: bool = False,
    shared_neg_sampling: bool = False,
    extra_hard_negatives: list[dict[str, list[str]]] | None = None,
) -> Dataset:
    """Build (query, document, label) pairs for reranker training.

    When ``do_merge_queries=True`` (default): duplicate query IDs are merged
    (GT skill_ids unioned) and GT filtering is complete — no GT can appear
    as a negative.

    When ``do_merge_queries=False``: duplicate rows are kept as-is, each row
    only knows its own skill_ids.  For multi-GT queries this means other GTs
    may appear as negatives (GT-as-negative leak).  This was the setting for
    the original SkillRet-Reranker-0.6B and produced more training data (~2x).

    When ``per_positive_negatives=False`` (default): samples ``num_negatives``
    negatives once per query, shared across all its positive pairs.

    When ``per_positive_negatives=True``: samples ``num_negatives`` negatives
    independently for *each* positive pair, producing ~2x more training data
    when queries have multiple GTs.

    When ``shared_neg_sampling=True`` (requires ``per_positive_negatives=True``):
    randomly samples ``num_negatives`` negatives *once per query* from the pool,
    then reuses that same set for every GT group of the query.  This combines
    random diversity (avoiding a homogeneous top-K cluster) with the 3x gradient
    reinforcement benefit of consistent negatives across GT groups.

    When ``extra_hard_negatives`` is provided (a list of additional hard-negative
    dicts from other embedding models), the candidates from each source are sliced
    with the same ``hard_neg_skip_top`` / ``hard_neg_top_k`` window independently,
    then unioned with the primary ``hard_negatives`` pool.  This gives a more
    diverse candidate pool that covers failures of multiple embedding models.
    """
    if do_merge_queries:
        queries = merge_queries(queries)
    else:
        logger.info(
            f"  Skipping merge_queries — keeping {len(queries):,} rows as-is "
            f"(GT-as-negative leak possible for multi-positive queries)"
        )
    all_skill_ids = list(skill_lookup.keys())
    rng = random.Random(seed)

    rows_query, rows_doc, rows_label, rows_group_id = [], [], [], []
    skipped = 0
    hard_used, random_used = 0, 0
    group_id = 0

    for q in queries:
        gt_ids = set(q["skill_ids"])
        qid = q["id"]

        hn_ids = []
        if hard_negatives and qid in hard_negatives:
            ranked = hard_negatives[qid]
            end_idx = hard_neg_top_k if hard_neg_top_k else len(ranked)
            ranked = ranked[hard_neg_skip_top:end_idx]
            hn_ids = [sid for sid in ranked
                      if sid not in gt_ids and sid in skill_lookup]

        # Union hard negatives from extra embedding sources (each sliced with the
        # same skip/top_k window, then deduplicated against the primary pool and GTs).
        if extra_hard_negatives:
            seen = set(hn_ids) | gt_ids
            for extra_hn in extra_hard_negatives:
                if qid not in extra_hn:
                    continue
                extra_ranked = extra_hn[qid]
                end_idx = hard_neg_top_k if hard_neg_top_k else len(extra_ranked)
                extra_ranked = extra_ranked[hard_neg_skip_top:end_idx]
                for sid in extra_ranked:
                    if sid not in seen and sid in skill_lookup:
                        hn_ids.append(sid)
                        seen.add(sid)

        # shared_neg_sampling: sample once per query, reuse for all GT groups.
        # Gives random diversity while preserving 3x gradient reinforcement.
        shared_negs: list[str] | None = None
        if shared_neg_sampling and per_positive_negatives and len(hn_ids) >= num_negatives:
            shared_negs = rng.sample(hn_ids, num_negatives)
            hard_used += len(shared_negs)

        if per_positive_negatives:
            for sid in q["skill_ids"]:
                skill_text = skill_lookup.get(sid)
                if skill_text is None:
                    skipped += 1
                    continue
                rows_query.append(q["query"])
                rows_doc.append(skill_text)
                rows_label.append(1)
                rows_group_id.append(group_id)

                neg_ids: list[str] = []
                if shared_negs is not None:
                    # Same randomly-sampled negatives for every GT of this query.
                    neg_ids = list(shared_negs)
                elif random_neg_sampling and len(hn_ids) > num_negatives:
                    # Random sample from the full pool — each positive gets a
                    # different draw, preventing identical negatives across
                    # groups of the same multi-positive query.
                    sampled = rng.sample(hn_ids, num_negatives)
                    neg_ids = sampled
                    hard_used += len(neg_ids)
                else:
                    for hn_sid in hn_ids:
                        if len(neg_ids) >= num_negatives:
                            break
                        neg_ids.append(hn_sid)
                        hard_used += 1

                while len(neg_ids) < num_negatives:
                    neg_id = rng.choice(all_skill_ids)
                    while neg_id in gt_ids or neg_id in neg_ids:
                        neg_id = rng.choice(all_skill_ids)
                    neg_ids.append(neg_id)
                    random_used += 1

                for neg_id in neg_ids:
                    rows_query.append(q["query"])
                    rows_doc.append(skill_lookup[neg_id])
                    rows_label.append(0)
                    rows_group_id.append(group_id)

                group_id += 1
        else:
            has_pos = False
            n_pos_this = 0
            for sid in q["skill_ids"]:
                skill_text = skill_lookup.get(sid)
                if skill_text is None:
                    skipped += 1
                    continue
                rows_query.append(q["query"])
                rows_doc.append(skill_text)
                rows_label.append(1)
                rows_group_id.append(group_id)
                has_pos = True
                n_pos_this += 1

            if not has_pos:
                continue

            # If fixed_group_size is set (e.g. 20 to match SkillRouter), reduce
            # num_negatives so that total group size = fixed_group_size regardless
            # of how many positives the query has.
            effective_neg = num_negatives
            if fixed_group_size is not None:
                effective_neg = max(1, fixed_group_size - n_pos_this)

            neg_ids = []
            if (shared_neg_sampling or random_neg_sampling) and len(hn_ids) >= effective_neg:
                neg_ids = rng.sample(hn_ids, effective_neg)
                hard_used += len(neg_ids)
            else:
                for hn_sid in hn_ids:
                    if len(neg_ids) >= effective_neg:
                        break
                    neg_ids.append(hn_sid)
                    hard_used += 1

            while len(neg_ids) < effective_neg:
                neg_id = rng.choice(all_skill_ids)
                while neg_id in gt_ids or neg_id in neg_ids:
                    neg_id = rng.choice(all_skill_ids)
                neg_ids.append(neg_id)
                random_used += 1

            for neg_id in neg_ids:
                rows_query.append(q["query"])
                rows_doc.append(skill_lookup[neg_id])
                rows_label.append(0)
                rows_group_id.append(group_id)

            group_id += 1

    n_pos = sum(1 for l in rows_label if l == 1)
    n_neg = sum(1 for l in rows_label if l == 0)
    neg_mode = "neg/positive" if per_positive_negatives else "neg/query"
    logger.info(
        f"Training pairs: {len(rows_query):,} "
        f"({n_pos:,} pos + {n_neg:,} neg, {num_negatives} {neg_mode}, "
        f"from {len(queries):,} queries, skipped {skipped})"
    )
    if hard_negatives:
        logger.info(
            f"  Hard negatives: skip_top={hard_neg_skip_top}, "
            f"top_k={hard_neg_top_k or 'all'}, "
            f"used={hard_used:,}, random_fill={random_used:,}"
        )
        if extra_hard_negatives:
            logger.info(
                f"  Multi-source pool: 1 primary + {len(extra_hard_negatives)} extra source(s)"
            )

    return Dataset.from_dict(
        {"query": rows_query, "document": rows_doc, "label": rows_label,
         "group_id": rows_group_id}
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data collator — tokenises with the Qwen3-Reranker chat template
# ═══════════════════════════════════════════════════════════════════════════

class RerankerCollator:
    """Tokenise (query, document, label) dicts into model-ready batches.

    Reproduces the exact input format used by Qwen3RankModel in eval.py:
        <|im_start|>system\n{SYSTEM}<|im_end|>\n
        <|im_start|>user\n
        <Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}
        <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    The model then predicts the next token -> "yes" or "no".
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        instruction: str = SKILL_RERANK_INSTRUCTION,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

        prefix = (
            f"<|im_start|>system\n{RERANKER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        self.max_content = max_length - len(self.prefix_ids) - len(self.suffix_ids)

    def _format_content(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def __call__(self, features: list[dict]) -> dict:
        input_ids_list = []
        labels = []

        for f in features:
            text = self._format_content(f["query"], f["document"])
            content_ids = self.tokenizer.encode(
                text, add_special_tokens=False,
            )[:self.max_content]
            ids = self.prefix_ids + content_ids + self.suffix_ids
            input_ids_list.append(ids)
            labels.append(f["label"])

        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        pad_id = self.tokenizer.pad_token_id

        padded_ids, attn_masks = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attn_masks.append([0] * pad_len + [1] * len(ids))

        result = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }
        if "group_id" in features[0]:
            result["group_ids"] = torch.tensor(
                [f["group_id"] for f in features], dtype=torch.long
            )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Custom Trainer — yes/no token BCE loss (matches Qwen3-Reranker eval)
# ═══════════════════════════════════════════════════════════════════════════

class RerankerTrainer(Trainer):
    """Trainer that computes BCE loss on P("yes") vs P("no") logits.

    At the last token position, extracts logits for the "yes" and "no"
    vocabulary tokens, computes log_softmax, and applies binary cross-
    entropy against the ground-truth label.  This is identical to how
    Qwen3RankModel scores at inference time.
    """

    def __init__(self, token_true_id: int, token_false_id: int, **kwargs):
        super().__init__(**kwargs)
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Skip Trainer eval_loss entirely — TrecEvalCallback computes all metrics.

        Only fires on_evaluate callbacks (which run TrecEvalCallback → NDCG).
        Nothing is logged to wandb from the Trainer side; our wandb.log() call
        in TrecEvalCallback handles all metric reporting.
        """
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics={}
        )
        return {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("group_ids", None)  # unused in pointwise; popped for listwise compat

        raw = model.module if hasattr(model, "module") else model
        base_out = raw.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        last_hidden = base_out.last_hidden_state[:, -1:, :]
        last_logits = raw.lm_head(last_hidden).squeeze(1)

        true_v = last_logits[:, self.token_true_id]
        false_v = last_logits[:, self.token_false_id]
        stacked = torch.stack([false_v, true_v], dim=1)
        log_probs = F.log_softmax(stacked, dim=1)
        p_yes = log_probs[:, 1].exp()

        loss = F.binary_cross_entropy(p_yes, labels.to(p_yes.dtype))
        if return_outputs:
            return loss, (p_yes.detach(),)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Listwise loss — GroupBatchSampler + ListwiseRerankerTrainer
# ═══════════════════════════════════════════════════════════════════════════

class GroupBatchSampler(torch.utils.data.Sampler):
    """Yields index batches where every batch contains only complete query groups.

    Groups are sharded across DDP ranks at the group level (not sample level),
    so no group is ever split across GPUs.  Within each rank, groups are
    shuffled per epoch and packed greedily into batches of ``batch_size``
    samples.

    Args:
        group_ids:  Per-sample group identifier (aligned with dataset indices).
        batch_size: Target number of *samples* per batch (not groups).
        shuffle:    Shuffle group order each epoch.
        seed:       Base RNG seed; epoch is added for per-epoch shuffling.
        rank:       DDP local rank (0-based).
        world_size: Total number of DDP processes.
        epoch:      Current epoch (call ``set_epoch`` before each epoch).
    """

    def __init__(
        self,
        group_ids: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
    ):
        from collections import defaultdict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch

        self._groups: dict[int, list[int]] = defaultdict(list)
        for i, gid in enumerate(group_ids):
            self._groups[gid].append(i)
        self._group_keys = list(self._groups.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        keys = self._group_keys.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(keys)

        # Shard groups across DDP ranks (interleaved so load is balanced)
        rank_keys = keys[self.rank::self.world_size]

        batch: list[int] = []
        for key in rank_keys:
            indices = self._groups[key]
            if batch and len(batch) + len(indices) > self.batch_size:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch

    def __len__(self) -> int:
        n_rank_groups = max(1, len(self._group_keys) // self.world_size)
        avg_size = sum(len(v) for v in self._groups.values()) / max(1, len(self._groups))
        return max(1, round(n_rank_groups * avg_size / self.batch_size))


class ListwiseRerankerTrainer(RerankerTrainer):
    """Trainer using listwise softmax cross-entropy loss.

    For each query group in the batch, computes the softmax over the
    log-odds ranking scores of all candidates and applies cross-entropy
    on the positive(s):

        score_i  = log P("yes"_i) - log P("no"_i)   [log-odds]
        log_prob = log_softmax(scores)               [over the group]
        loss     = -mean(log_prob[positive_indices]) [avg over positives]

    Uses GroupBatchSampler to keep complete groups within each device batch.
    The eval_loss (used for best-checkpoint selection) falls back to BCE so
    the flat eval dataset does not need group_id.
    """

    def _make_group_dataloader(self, dataset, shuffle: bool, epoch: int = 0):
        """Build a DataLoader that keeps each query group in the same batch."""
        from torch.utils.data import DataLoader
        group_ids: list[int] = dataset["group_id"]
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        sampler = GroupBatchSampler(
            group_ids=group_ids,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            seed=self.args.seed,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def get_train_dataloader(self):
        epoch = int(getattr(self.state, "epoch", 0)) if hasattr(self, "state") else 0
        return self._make_group_dataloader(self.train_dataset, shuffle=True, epoch=epoch)

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if ds is not None and "group_id" in ds.column_names:
            return self._make_group_dataloader(ds, shuffle=False, epoch=0)
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        group_ids = inputs.pop("group_ids", None)

        raw = model.module if hasattr(model, "module") else model
        base_out = raw.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        last_hidden = base_out.last_hidden_state[:, -1:, :]
        last_logits = raw.lm_head(last_hidden).squeeze(1)

        true_v = last_logits[:, self.token_true_id]
        false_v = last_logits[:, self.token_false_id]

        if group_ids is None:
            # Eval path (flat dataset, no group_ids) — fall back to BCE
            stacked = torch.stack([false_v, true_v], dim=1)
            p_yes = F.log_softmax(stacked, dim=1)[:, 1].exp()
            loss = F.binary_cross_entropy(p_yes, labels.to(p_yes.dtype))
            if return_outputs:
                return loss, (p_yes.detach(),)
            return loss

        # Listwise: log-odds as ranking score, softmax CE per group
        ranking_scores = true_v - false_v  # (batch,)

        unique_groups = torch.unique(group_ids)
        group_losses: list[torch.Tensor] = []
        for gid in unique_groups:
            mask = group_ids == gid
            g_scores = ranking_scores[mask]
            g_labels = labels[mask]
            pos_mask = g_labels.bool()
            if not pos_mask.any():
                continue
            # Multi-positive distribution: uniform mass over all positives, 0 for negatives.
            # Using one-hot CE per positive (old approach) hurts because positives inflate
            # each other's denominator. The correct target is a soft distribution.
            target = g_labels.float() / g_labels.sum()
            log_probs = F.log_softmax(g_scores, dim=0)
            group_losses.append(-(target * log_probs).sum())

        if not group_losses:
            loss = ranking_scores.sum() * 0.0  # differentiable zero
        else:
            loss = torch.stack(group_losses).mean()

        if return_outputs:
            stacked = torch.stack([false_v, true_v], dim=1)
            p_yes = F.log_softmax(stacked, dim=1)[:, 1].exp()
            return loss, (p_yes.detach(),)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# TREC-eval callback — proper NDCG/MAP/Recall via pytrec_eval
# ═══════════════════════════════════════════════════════════════════════════

class TrecEvalCallback(TrainerCallback):
    """Evaluate reranking quality using trec_eval from skillret/eval.py.

    Builds a candidate pool per query (ground-truth positives + top-k from
    first-stage retrieval), scores all pairs with the model's yes/no logits,
    then feeds qrels + results into ``trec_eval`` (pytrec_eval).
    """

    def __init__(
        self,
        first_stage_file: str,
        collator: RerankerCollator,
        token_true_id: int,
        token_false_id: int,
        batch_size: int = 16,
        from_top_k: int = 20,
        max_queries: int = 500,
        seed: int = 43,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.001,
        eval_steps: int = 100,
    ):
        self.collator = collator
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self._es_best_ndcg = -1.0
        self._es_counter = 0
        # Best-checkpoint tracking (independent of early stopping). Updated on
        # every eval so that final/ can be saved from the best step rather than
        # the last (often overfit) step. See best-checkpoint save in main().
        self.best_ndcg = -1.0
        self.best_step: int | None = None

        queries = load_eval_queries()
        skills = load_eval_skills()
        skill_lookup = build_skill_lookup(skills)

        rng_sample = random.Random(seed)
        if max_queries and len(queries) > max_queries:
            queries = rng_sample.sample(queries, max_queries)

        with open(first_stage_file) as f:
            fs_data = json.load(f)
        fs_retrieval = fs_data.get("retrieval", fs_data)
        fs_results = list(fs_retrieval.values())[0] if fs_retrieval else {}

        self.qrels: dict[str, dict[str, int]] = {}
        self.eval_samples: list[dict] = []

        for q in queries:
            qid = q["id"]
            if qid not in fs_results:
                continue

            if "labels" in q:
                raw_labels = q["labels"]
                if isinstance(raw_labels, str):
                    raw_labels = json.loads(raw_labels)
                self.qrels[qid] = {
                    str(x["id"]): int(x["relevance"]) for x in raw_labels
                }
            elif "skill_ids" in q:
                self.qrels[qid] = {str(sid): 1 for sid in q["skill_ids"]}
            else:
                continue

            ranked = sorted(
                fs_results[qid].items(), key=lambda x: x[1], reverse=True
            )[:from_top_k]
            candidate_ids = [sid for sid, _ in ranked if sid in skill_lookup]
            if not candidate_ids:
                continue

            self.eval_samples.append({
                "qid": qid,
                "query": q["query"],
                "candidate_ids": candidate_ids,
                "candidate_texts": [skill_lookup[sid] for sid in candidate_ids],
            })

        logger.info(
            f"  TrecEvalCallback: {len(self.eval_samples)} queries, "
            f"top-{from_top_k} from first-stage ({Path(first_stage_file).name})"
        )

    def to_dataset(self) -> Dataset:
        """Convert the eval samples into a Dataset for Trainer eval loss.

        Includes group_id so ListwiseRerankerTrainer computes listwise CE
        (instead of falling back to BCE) during evaluation.
        """
        rows_q, rows_d, rows_l, rows_g = [], [], [], []
        for gid, s in enumerate(self.eval_samples):
            qrels_for_q = self.qrels.get(s["qid"], {})
            for sid, text in zip(s["candidate_ids"], s["candidate_texts"]):
                rows_q.append(s["query"])
                rows_d.append(text)
                rows_l.append(1 if qrels_for_q.get(str(sid), 0) > 0 else 0)
                rows_g.append(gid)
        return Dataset.from_dict(
            {"query": rows_q, "document": rows_d, "label": rows_l, "group_id": rows_g}
        )

    @torch.no_grad()
    def _score_pairs(self, model, pairs: list[dict]) -> list[float]:
        # Returns log-odds (true_v - false_v) per pair: ranks identically to
        # p_yes for trec_eval (monotonic) and feeds listwise CE eval loss.
        raw = model.module if hasattr(model, "module") else model
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = self.collator(pairs[i:i + self.batch_size])
            batch = {k: v.to(raw.device) for k, v in batch.items()}
            batch.pop("labels")
            base_out = raw.model(**batch)
            last_hidden = base_out.last_hidden_state[:, -1:, :]
            last_logits = raw.lm_head(last_hidden).squeeze(1)
            true_v = last_logits[:, self.token_true_id]
            false_v = last_logits[:, self.token_false_id]
            scores.extend((true_v - false_v).float().cpu().tolist())
        return scores

    def _evaluate(self, model) -> dict[str, float]:
        from skillret.eval import trec_eval

        results: dict[str, dict[str, float]] = {}
        for s in self.eval_samples:
            pairs = [{"query": s["query"], "document": d, "label": 0}
                     for d in s["candidate_texts"]]
            scores = self._score_pairs(model, pairs)
            results[s["qid"]] = {
                str(sid): float(sc)
                for sid, sc in zip(s["candidate_ids"], scores)
            }

        return trec_eval(qrels=self.qrels, results=results)

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger evaluation at eval_steps intervals (since eval_strategy='no')."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        model.eval()

        # Sync all ranks before scoring — ensures all ranks start scoring at the same
        # time regardless of how long the Trainer's distributed eval_loss took per rank.
        # Since shards are equal-sized (4997//8 = 624-625 queries), all ranks finish
        # scoring at roughly the same time, so all_gather_object completes in seconds.
        if is_dist:
            logger.info(f"[TrecEval rank {rank}] waiting at barrier before scoring...")
            dist.barrier()
            logger.info(f"[TrecEval rank {rank}] barrier passed, starting scoring ({len(self.eval_samples) // world_size + (1 if rank < len(self.eval_samples) % world_size else 0)} queries)")

        from tqdm import tqdm as _tqdm
        if world_size > 1:
            shard = self.eval_samples[rank::world_size]
            local_results: dict[str, dict[str, float]] = {}
            for i, s in enumerate(shard):
                if i % 100 == 0:
                    logger.info(f"[TrecEval rank {rank}] scoring {i}/{len(shard)} queries...")
                pairs = [{"query": s["query"], "document": d, "label": 0}
                         for d in s["candidate_texts"]]
                scores = self._score_pairs(model, pairs)
                local_results[s["qid"]] = {
                    str(sid): float(sc)
                    for sid, sc in zip(s["candidate_ids"], scores)
                }
            logger.info(f"[TrecEval rank {rank}] scoring done, calling all_gather_object...")
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_results)
            if rank != 0:
                # Wait for rank 0 to broadcast the early-stopping decision, then exit.
                stop_flag = [False]
                dist.broadcast_object_list(stop_flag, src=0)
                control.should_training_stop = stop_flag[0]
                return
            full_results: dict[str, dict[str, float]] = {}
            for r in gathered:
                full_results.update(r)
        else:
            full_results = {}
            for s in _tqdm(self.eval_samples, desc="TrecEval scoring", leave=False):
                pairs = [{"query": s["query"], "document": d, "label": 0}
                         for d in s["candidate_texts"]]
                scores = self._score_pairs(model, pairs)
                full_results[s["qid"]] = {
                    str(sid): float(sc)
                    for sid, sc in zip(s["candidate_ids"], scores)
                }

        from skillret.eval import trec_eval
        metrics = trec_eval(qrels=self.qrels, results=full_results)

        # Listwise CE over the full benchmark, reusing the same scores we
        # ranked. Per-query mean (each query weighted equally — matches NDCG
        # averaging). Replaces the brittle Phase-1 eval_loss path.
        total_loss, n_queries = 0.0, 0
        for s in self.eval_samples:
            qid = s["qid"]
            if qid not in full_results:
                continue
            qrels_for_q = self.qrels.get(qid, {})
            cand_ids = s["candidate_ids"]
            cand_scores = torch.tensor(
                [full_results[qid][str(sid)] for sid in cand_ids],
                dtype=torch.float32,
            )
            cand_labels = torch.tensor(
                [1.0 if qrels_for_q.get(str(sid), 0) > 0 else 0.0
                 for sid in cand_ids],
                dtype=torch.float32,
            )
            if cand_labels.sum() == 0:
                continue
            target = cand_labels / cand_labels.sum()
            log_probs = F.log_softmax(cand_scores, dim=0)
            total_loss += -(target * log_probs).sum().item()
            n_queries += 1
        eval_loss = total_loss / n_queries if n_queries > 0 else 0.0

        ndcg10 = metrics.get("NDCG@10", 0.0)
        map10 = metrics.get("MAP@10", 0.0)
        recall10 = metrics.get("Recall@10", 0.0)
        logger.info(
            f"[Eval step {state.global_step}] benchmark  "
            f"loss={eval_loss:.4f}  NDCG@10={ndcg10:.4f}  "
            f"MAP@10={map10:.4f}  Recall@10={recall10:.4f}"
        )

        # Track the best step by NDCG@10 (rank 0 only, where full_results live)
        # so main() can save final/ from the best checkpoint instead of the last.
        if ndcg10 > self.best_ndcg:
            self.best_ndcg = ndcg10
            self.best_step = state.global_step
            logger.info(f"[BestCkpt] new best NDCG@10={ndcg10:.4f} at step {state.global_step}")

        if state.log_history is not None:
            log_entry = {
                "step": state.global_step,
                "eval_benchmark_loss": eval_loss,
            }
            for k, v in metrics.items():
                log_entry[f"eval_benchmark_{k.lower()}"] = v
            state.log_history.append(log_entry)

        try:
            import wandb
            if wandb.run is not None:
                wandb_log = {"eval/benchmark_loss": eval_loss, "eval/global_step": state.global_step}
                for k, v in metrics.items():
                    wandb_log[f"eval/{k.replace('@', '_at_').lower()}"] = v
                wandb.log(wandb_log)
        except Exception as e:
            logger.warning(f"[TrecEval rank 0] wandb.log failed: {e}")

        if self.early_stopping_patience > 0:
            if ndcg10 > self._es_best_ndcg + self.early_stopping_min_delta:
                self._es_best_ndcg = ndcg10
                self._es_counter = 0
            else:
                self._es_counter += 1
                logger.info(
                    f"[EarlyStopping] No improvement ≥{self.early_stopping_min_delta} "
                    f"({self._es_counter}/{self.early_stopping_patience}), "
                    f"best NDCG@10={self._es_best_ndcg:.4f}"
                )
                if self._es_counter >= self.early_stopping_patience:
                    logger.info(
                        f"[EarlyStopping] Patience exhausted — stopping training at step {state.global_step}. "
                        f"Best NDCG@10={self._es_best_ndcg:.4f}"
                    )
                    control.should_training_stop = True

        # Broadcast should_training_stop from rank 0 to all ranks so every DDP
        # process stops cleanly without hitting NCCL collective timeouts.
        if is_dist:
            stop_flag = [control.should_training_stop]
            dist.broadcast_object_list(stop_flag, src=0)
            control.should_training_stop = stop_flag[0]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-Reranker")
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--mine-hard-negatives", action="store_true",
        help="Only mine hard negatives and save to file, then exit.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in output_dir.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        fh = logging.FileHandler(
            Path(cfg["output_dir"]) / "train.log", mode="a"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)

    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # ── Hard-negative mining mode ──────────────────────────────────────────
    if args.mine_hard_negatives:
        emb_model = cfg.get("embedding_model")
        if not emb_model:
            raise ValueError(
                "Set 'embedding_model' in config for hard-negative mining "
                "(e.g. Thaki/SkillRet-Embedding-8B)"
            )
        model_base = os.environ.get("MODEL_BASE_DIR", "")
        if model_base:
            local_path = Path(model_base) / emb_model
            if local_path.exists():
                logger.info(f"Using local model: {local_path}")
                emb_model = str(local_path)
        hn_path = cfg.get("hard_negatives_file")
        if not hn_path:
            hn_path = str(PROJECT_ROOT / "data" / "hard_negatives.json")
        mine_hard_negatives(
            embedding_model=emb_model,
            keep_top=cfg.get("mine_keep_top", 100),
            output_file=Path(hn_path),
            encode_batch_size=cfg.get("mine_encode_batch_size", 32),
        )
        return

    # ── Training mode ──────────────────────────────────────────────────────
    logger.info(f"Loading model: {cfg['model']}")
    logger.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], padding_side="left", trust_remote_code=True,
    )
    logger.info("  Tokenizer loaded. Loading model weights (may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    logger.info("  Model loaded.")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    logger.info(f"  Token IDs — yes: {token_true_id}, no: {token_false_id}")

    collator = RerankerCollator(
        tokenizer=tokenizer,
        max_length=cfg["max_seq_length"],
    )
    logger.info(
        f"  max_seq_length={cfg['max_seq_length']}, "
        f"prefix={len(collator.prefix_ids)} toks, "
        f"suffix={len(collator.suffix_ids)} toks, "
        f"max_content={collator.max_content} toks"
    )

    # ── Data ───────────────────────────────────────────────────────────────
    logger.info("Loading training data from HuggingFace...")
    logger.info("  Downloading/loading skills dataset...")
    train_skills = load_train_skills()
    train_skill_lookup = build_skill_lookup(train_skills)
    logger.info(f"  {len(train_skill_lookup):,} skills loaded")

    hard_negatives = None
    hn_file = cfg.get("hard_negatives_file")
    if hn_file and Path(hn_file).exists():
        logger.info(f"Loading hard negatives from {hn_file}...")
        with open(hn_file) as f:
            hard_negatives = json.load(f)
        logger.info(f"  Hard negatives for {len(hard_negatives):,} queries")

    extra_hard_negatives: list[dict] = []
    hn_files = cfg.get("hard_negatives_files")
    if hn_files:
        # First file becomes the primary source; the rest are extra sources.
        for i, path in enumerate(hn_files):
            if not Path(path).exists():
                logger.warning(f"Hard negatives file not found, skipping: {path}")
                continue
            logger.info(f"Loading hard negatives [{i}] from {path}...")
            with open(path) as f:
                hn_data = json.load(f)
            logger.info(f"  Hard negatives for {len(hn_data):,} queries")
            if hard_negatives is None:
                hard_negatives = hn_data
            else:
                extra_hard_negatives.append(hn_data)
        if extra_hard_negatives:
            logger.info(
                f"Multi-source hard negatives: 1 primary + {len(extra_hard_negatives)} extra source(s)"
            )

    logger.info("  Downloading/loading queries dataset...")
    train_queries = load_train_queries()
    logger.info(f"  {len(train_queries):,} queries loaded")
    logger.info("Building training dataset...")
    train_dataset = build_train_dataset(
        train_queries,
        train_skill_lookup,
        num_negatives=cfg["num_negatives"],
        hard_negatives=hard_negatives,
        hard_neg_skip_top=cfg["hard_neg_skip_top"],
        hard_neg_top_k=cfg["hard_neg_top_k"],
        seed=seed,
        per_positive_negatives=cfg["per_positive_negatives"],
        do_merge_queries=cfg["merge_queries"],
        fixed_group_size=cfg["fixed_group_size"],
        random_neg_sampling=cfg.get("random_neg_sampling", False),
        shared_neg_sampling=cfg.get("shared_neg_sampling", False),
        extra_hard_negatives=extra_hard_negatives if extra_hard_negatives else None,
    )

    # ── Evaluator ──────────────────────────────────────────────────────────
    fs_file = cfg["eval_first_stage_file"]
    if not Path(fs_file).is_absolute():
        fs_file = str(PROJECT_ROOT / fs_file)
    eval_max_queries = cfg.get("eval_max_queries", 500)
    eval_label = f"{eval_max_queries} query sample" if eval_max_queries else "full eval set"
    logger.info(f"Building trec_eval evaluator (first-stage: {Path(fs_file).name}, {eval_label})...")
    eval_batch_size = cfg.get("eval_batch_size", 16)
    es_patience = cfg.get("early_stopping_patience", 0)
    es_min_delta = cfg.get("early_stopping_min_delta", 0.001)
    if es_patience > 0:
        logger.info(
            f"Early stopping enabled: patience={es_patience} eval steps, "
            f"min_delta={es_min_delta} NDCG@10"
        )
    eval_callback = TrecEvalCallback(
        first_stage_file=fs_file,
        collator=collator,
        token_true_id=token_true_id,
        token_false_id=token_false_id,
        batch_size=eval_batch_size,
        from_top_k=20,
        max_queries=eval_max_queries,
        seed=seed,
        early_stopping_patience=es_patience,
        early_stopping_min_delta=es_min_delta,
        eval_steps=cfg["eval_steps"],
    )

    eval_dataset = eval_callback.to_dataset()
    logger.info(f"Eval dataset (benchmark): {len(eval_dataset):,} pairs")

    # ── Training args ──────────────────────────────────────────────────────
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))

    wandb_run = cfg.get("wandb_run")
    if not wandb_run:
        model_name = Path(cfg["model"]).name
        neg_tag = "hn" if hard_negatives else "rn"
        wandb_run = f"{model_name}-{n_gpu}gpu-{neg_tag}{cfg['num_negatives']}-sft"

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=False,
        bf16=cfg["bf16"],
        ddp_find_unused_parameters=False,
        eval_strategy="no",   # TrecEvalCallback triggers eval via on_step_end
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit", 300),
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=False,  # eval runs in TrecEvalCallback; final/ is saved from the best NDCG@10 checkpoint in main()
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=wandb_run,
        seed=seed,
        remove_unused_columns=False,
    )

    loss_type = cfg.get("loss_type", "pointwise")
    TrainerClass = ListwiseRerankerTrainer if loss_type == "listwise" else RerankerTrainer
    logger.info(f"Loss type: {loss_type} → using {TrainerClass.__name__}")

    trainer = TrainerClass(
        token_true_id=token_true_id,
        token_false_id=token_false_id,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # evaluate() is overridden to skip eval_loss; TrecEvalCallback handles all evaluation
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )

    effective_batch = batch_size * n_gpu * grad_accum
    resume_ckpt = None
    if args.resume:
        ckpts = sorted(Path(cfg["output_dir"]).glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            logger.info(f"Resuming from {resume_ckpt}")
        else:
            logger.warning("--resume specified but no checkpoints found, starting fresh")
    logger.info(
        f"Starting training ({n_gpu} GPU DDP, effective_batch={effective_batch})..."
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if rank == 0:
        final_dir = Path(cfg["output_dir"]) / "final"

        # Determine the best step by NDCG@10. Prefer trainer.state.log_history
        # (persisted across resumes) so the best is found even if the run was
        # resumed after the peak; fall back to the callback's in-memory tracker.
        best_step, best_ndcg = None, -1.0
        for h in trainer.state.log_history:
            v = h.get("eval_benchmark_ndcg@10")
            if v is not None and v > best_ndcg:
                best_ndcg, best_step = v, h.get("step")
        if best_step is None and eval_callback.best_step is not None:
            best_step, best_ndcg = eval_callback.best_step, eval_callback.best_ndcg

        best_ckpt = (Path(cfg["output_dir"]) / f"checkpoint-{best_step}"
                     if best_step is not None else None)
        if best_ckpt is not None and best_ckpt.is_dir():
            import shutil
            logger.info(
                f"Saving BEST model to {final_dir} "
                f"(step {best_step}, NDCG@10={best_ndcg:.4f}) from {best_ckpt}..."
            )
            final_dir.mkdir(parents=True, exist_ok=True)
            # Copy model + tokenizer files, excluding optimizer/scheduler/RNG/
            # trainer state so final/ is a clean inference checkpoint.
            skip = {"optimizer.pt", "scheduler.pt", "scaler.pt",
                    "trainer_state.json", "training_args.bin"}
            for f in best_ckpt.iterdir():
                if f.is_dir() or f.name in skip or f.name.startswith("rng_state"):
                    continue
                shutil.copy2(f, final_dir / f.name)
            logger.info(f"Done! (final/ = best checkpoint-{best_step})")
        else:
            logger.warning(
                f"No best checkpoint found (best_step={best_step}); "
                f"saving last in-memory model to {final_dir} instead."
            )
            raw = model.module if hasattr(model, "module") else model
            raw.save_pretrained(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))
            logger.info("Done!")


if __name__ == "__main__":
    main()
