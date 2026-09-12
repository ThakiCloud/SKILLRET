"""Evaluation pipeline for SkillRet: embedding retrieval, BM25, and reranking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np
import pytrec_eval
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from ._compat import (
    apply_transformers_compat_patches,
    fix_non_persistent_buffers,
    fix_rotary_embeddings,
    patch_jina_v4_compat,
    reload_safetensors_weights,
)
from .config import (
    DEFAULT_RERANK_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    HF_DATASET_ID,
    RERANK_TOP_K,
    SKILL_RERANK_INSTRUCTION,
    get_batch_size,
    get_max_seq_length,
    get_st_config,
)
from .utils import load_json, write_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

apply_transformers_compat_patches()


# ---------------------------------------------------------------------------
# Data loading (HuggingFace datasets)
# ---------------------------------------------------------------------------

def _dataset_revision(revision: str | None = None) -> str | None:
    """Which Hub revision to pull.

    The Hub head is mutable, so an unpinned evaluation is not reproducible: the
    numbers you get depend on when you ran it. Pass ``revision`` explicitly, or
    set ``SKILLRET_DATASET_REVISION``, to pin it. Default stays unpinned so that
    existing callers keep their current behaviour.
    """
    return revision or os.getenv("SKILLRET_DATASET_REVISION") or None


def _load_hf_dataset(
    subset: str, split: str = "test", revision: str | None = None
) -> list[dict]:
    """Load a subset/split from HuggingFace, caching automatically."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
    ds = load_dataset(
        HF_DATASET_ID, subset, split=split, revision=_dataset_revision(revision)
    )
    return [dict(row) for row in ds]


def build_skill_text(skill: dict) -> str:
    """Build ``name | description | skill_md`` (full text, no truncation)."""
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    body = (skill.get("skill_md") or "").strip()
    return f"{name} | {desc} | {body}"


def _embedding_text_for_skill(skill: dict) -> str:
    """Text to embed for one skill (composite: name | description | skill_md)."""
    return build_skill_text(skill)


def load_corpus(split: str = "test", revision: str | None = None) -> list[dict]:
    """Load skill corpus from HuggingFace."""
    skills = _load_hf_dataset("skills", split=split, revision=revision)

    for s in skills:
        if "skill_md" not in s or not (s.get("skill_md") or "").strip():
            s["skill_md"] = s.get("description", "")
    return skills


def load_queries(split: str = "test", revision: str | None = None) -> list[dict]:
    """Load queries from HuggingFace."""
    return _load_hf_dataset("queries", split=split, revision=revision)


def _normalize_query_labels(item: dict) -> list[dict]:
    """Handle both 'labels' format and 'skill_ids' format for ground truth."""
    if "labels" in item:
        labels = item["labels"]
        if isinstance(labels, str):
            labels = json.loads(labels)
        return labels
    if "skill_ids" in item:
        return [{"id": sid, "relevance": 1} for sid in item["skill_ids"]]
    raise KeyError(f"Query {item.get('id')} has neither 'labels' nor 'skill_ids'")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _env_trust_remote_code() -> bool:
    """HF remote code execution; set SKILLRET_TRUST_REMOTE_CODE=0 to disable."""
    v = os.getenv("SKILLRET_TRUST_REMOTE_CODE", "true").strip().lower()
    return v not in ("0", "false", "no")


def _behavior_key(model_ref: str) -> str:
    """String used for query templates and pooling heuristics."""
    return model_ref



def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model_tokenizer(
    model_path: str,
    *,
    behavior_key: str,
    trust_remote_code: bool = True,
):
    """Load an embedding model, returning (model, tokenizer, is_sentence_transformer)."""
    fa2 = _has_flash_attn()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Native loading branches for models requiring special handling ---

    if "SkillRouter" in behavior_key:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=_token, trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        kwargs = {"torch_dtype": torch.bfloat16}
        if fa2:
            kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, token=_token, **kwargs,
        )
        model = model.to(dev).eval()
        return model, tokenizer, False

    if "NV-Embed" in behavior_key:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, token=_token,
        )
        model = model.to(dev).eval()
        return model, model.tokenizer, False

    if "jina-embeddings-v4" in behavior_key:
        patch_jina_v4_compat(model_path)

    if "inf-retriever" in behavior_key:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=_token, trust_remote_code=trust_remote_code,
        )
        kwargs = {}
        if fa2:
            kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16, token=_token, **kwargs,
        )
        reload_safetensors_weights(model, model_path)
        model = model.to(dev).eval()
        return model, tokenizer, False

    if "KaLM" in behavior_key:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=_token, trust_remote_code=trust_remote_code,
        )
        kwargs = {"torch_dtype": torch.bfloat16}
        if fa2:
            kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, token=_token, **kwargs,
        )
        model = model.to(dev).eval()
        return model, tokenizer, False

    # --- SentenceTransformer loading (default) ---

    model_kwargs: dict = {"token": _token, "torch_dtype": "auto"}
    config_kwargs: dict = {}

    model = SentenceTransformer(
        model_path,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
        config_kwargs=config_kwargs,
    )
    fix_non_persistent_buffers(model)

    _cfg_max = get_max_seq_length(model_path, mode="embed")
    if _cfg_max and model.max_seq_length != _cfg_max:
        model.max_seq_length = _cfg_max

    try:
        tokenizer = model.tokenizer
    except (AttributeError, RuntimeError):
        first_mod = next(iter(model.modules()))
        if hasattr(first_mod, "processor") and hasattr(first_mod.processor, "tokenizer"):
            tokenizer = first_mod.processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, token=_token, trust_remote_code=trust_remote_code,
            )
    return model, tokenizer, True


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

def _encode_native_lasttoken(
    model, tokenizer, texts: list[str], batch_size: int, max_length: int = 32768,
) -> np.ndarray:
    """Encode texts using last-token pooling + L2 normalize (native transformers)."""
    import torch.nn.functional as F
    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="native-encode"):
        batch = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        attn = batch_dict["attention_mask"]
        last_hidden = outputs.last_hidden_state
        left_padding = attn[:, -1].sum() == attn.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            seq_lengths = attn.sum(dim=1) - 1
            emb = last_hidden[
                torch.arange(last_hidden.shape[0], device=last_hidden.device),
                seq_lengths,
            ]
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Query / passage formatting
# ---------------------------------------------------------------------------

def _format_query(behavior_key: str, query: str) -> str:
    cfg = get_st_config(behavior_key)
    return cfg.get("query_prefix", "") + query


def _format_passage(behavior_key: str, passage: str) -> str:
    cfg = get_st_config(behavior_key)
    return cfg.get("doc_prefix", "") + passage


# ---------------------------------------------------------------------------
# RetModel: unified embedding model interface
# ---------------------------------------------------------------------------

class RetModel:
    """Unified interface for encoding queries and corpus documents."""

    def __init__(
        self,
        model_path: str,
        *,
        trust_remote_code: bool | None = None,
        behavior_key: str | None = None,
    ):
        self.model_path = model_path
        self.behavior_key = behavior_key or _behavior_key(model_path)
        trc = _env_trust_remote_code() if trust_remote_code is None else trust_remote_code
        self.model, self.tokenizer, self.st = _load_model_tokenizer(
            model_path,
            behavior_key=self.behavior_key,
            trust_remote_code=trc,
        )
        self._max_seq_length = get_max_seq_length(model_path, mode="embed")

    def encode_queries(self, queries: list[dict], batch_size: int) -> np.ndarray:
        _ml = self._max_seq_length or 32768
        bk = self.behavior_key

        # Native last-token models
        if "inf-retriever" in bk:
            texts = [
                f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q['query']}"
                for q in queries
            ]
            return _encode_native_lasttoken(self.model, self.tokenizer, texts, batch_size, max_length=_ml)

        if "NV-Embed" in bk:
            texts = [q["query"] for q in queries]
            all_embs = []
            n_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="NV-Embed queries"):
                batch = texts[i : i + batch_size]
                emb = self.model.encode(
                    batch,
                    instruction="Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
                    max_length=_ml,
                )
                all_embs.append(emb.cpu().detach().numpy().astype(np.float32))
            return np.concatenate(all_embs, axis=0)

        if "SkillRouter" in bk or "KaLM" in bk:
            cfg = get_st_config(bk)
            prompt = cfg.get("query_kwargs", {}).get("prompt", "")
            texts = [prompt + q["query"] for q in queries]
            return _encode_native_lasttoken(self.model, self.tokenizer, texts, batch_size, max_length=_ml)

        # SentenceTransformer path
        texts = [_format_query(bk, q["query"]) for q in queries]
        if self.st:
            cfg = get_st_config(bk)
            query_kwargs = cfg.get("query_kwargs", {})
            encode_kwargs: dict = {"batch_size": batch_size}
            encode_kwargs.update(query_kwargs)
            if "task" in query_kwargs or "prompt_name" in query_kwargs:
                embeddings = self.model.encode(texts, **encode_kwargs)
            else:
                embeddings = self.model.encode_query(texts, **encode_kwargs)
        else:
            raise ValueError(f"No query encoding strategy for model: {bk!r}")
        return np.asarray(embeddings, dtype=np.float32)

    def encode_corpus(
        self,
        skills: list[dict],
        batch_size: int,
    ) -> np.ndarray:
        bk = self.behavior_key
        texts = [
            _format_passage(bk, _embedding_text_for_skill(skill))
            for skill in skills
        ]
        _ml = self._max_seq_length or 32768

        # Native last-token models
        if "inf-retriever" in bk or "SkillRouter" in bk or "KaLM" in bk:
            return _encode_native_lasttoken(self.model, self.tokenizer, texts, batch_size, max_length=_ml)

        if "NV-Embed" in bk:
            all_embs = []
            n_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="NV-Embed corpus"):
                batch = texts[i : i + batch_size]
                emb = self.model.encode(batch, instruction="", max_length=_ml)
                all_embs.append(emb.cpu().detach().numpy().astype(np.float32))
            return np.concatenate(all_embs, axis=0)

        # SentenceTransformer path
        if self.st:
            cfg = get_st_config(bk)
            doc_kwargs = cfg.get("doc_kwargs", {})
            encode_kwargs: dict = {"batch_size": batch_size, "show_progress_bar": True}
            encode_kwargs.update(doc_kwargs)
            if "task" in doc_kwargs or "prompt_name" in doc_kwargs:
                emb = self.model.encode(texts, **encode_kwargs)
            else:
                emb = self.model.encode_document(texts, **encode_kwargs)
        else:
            raise ValueError(f"No corpus encoding strategy for model: {bk!r}")
        return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# TREC-style evaluation
# ---------------------------------------------------------------------------

def trec_eval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (5, 10, 15),
) -> Dict[str, float]:
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    prec = {f"Precision@{k}": 0.0 for k in k_values}
    comp = {f"Completeness@{k}": 0.0 for k in k_values}

    map_string = "map_cut." + ",".join(str(k) for k in k_values)
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_string = "recall." + ",".join(str(k) for k in k_values)
    precision_string = "P." + ",".join(str(k) for k in k_values)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += scores[query_id][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
            prec[f"Precision@{k}"] += scores[query_id][f"P_{k}"]
            comp[f"Completeness@{k}"] += float(
                scores[query_id][f"recall_{k}"] == 1.0
            )

    n = max(len(scores), 1)
    normalize = lambda m: {k: round(v / n, 5) for k, v in m.items()}

    all_metrics: Dict[str, float] = {}
    for m in [normalize(ndcg), normalize(_map), normalize(recall), normalize(prec), normalize(comp)]:
        all_metrics.update(m)
    return all_metrics


# ---------------------------------------------------------------------------
# FAISS corpus cache
# ---------------------------------------------------------------------------

_CACHE_VERSION = 1


def _embedding_cache_name(model_ref: str) -> str:
    """Human-readable cache filename stem from the model path or HF ID.

    Extracts the last two path components (org/model) and joins with '--'.
    E.g. '/DATA2/models/BAAI/bge-large-en-v1.5' -> 'BAAI--bge-large-en-v1.5'
         'BAAI/bge-large-en-v1.5' -> 'BAAI--bge-large-en-v1.5'
    """
    parts = Path(model_ref).parts
    # Use last two components if available (org/model), otherwise just the name
    key_parts = parts[-2:] if len(parts) >= 2 else parts[-1:]
    return "--".join(key_parts)


def _embedding_cache_paths(cache_dir: Path, name: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.faiss", cache_dir / f"{name}.meta.json"


def _try_load_embedding_index(
    *,
    faiss_index_path: Path,
    meta_path: Path,
    model_ref: str,
    n_skills: int,
):
    if not faiss_index_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("version") != _CACHE_VERSION:
        return None
    if meta.get("model_ref") != model_ref:
        return None
    if meta.get("n_skills") != n_skills:
        return None
    try:
        index = faiss.read_index(str(faiss_index_path))
    except RuntimeError:
        return None
    if index.ntotal != n_skills:
        return None
    return index


def _save_embedding_index(
    *,
    faiss_index_path: Path,
    meta_path: Path,
    model_ref: str,
    n_skills: int,
    dim: int,
    index,
) -> None:
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))
    write_json(
        {
            "version": _CACHE_VERSION,
            "model_ref": model_ref,
            "dataset": HF_DATASET_ID,
            "n_skills": n_skills,
            "dim": dim,
            "metric": "METRIC_INNER_PRODUCT",
            "index_type": "Flat",
        },
        meta_path,
    )


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def eval_retrieval(
    model_path: str,
    batch_size: int = 0,
    top_k: int = RERANK_TOP_K,
    output_file: str | None = None,
    split: str = "test",
    skill_ids_filter: set | None = None,
    trust_remote_code: bool | None = None,
    embedding_cache_dir: str | Path | None = None,
    use_embedding_cache: bool = True,
    force_rebuild_embedding_cache: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate an embedding model on skill retrieval.

    Args:
        model_path: HuggingFace model ID (e.g. ``"BAAI/bge-small-en-v1.5"``)
            or local directory with a checkpoint.
        batch_size: Encoding batch size. 0 = auto-detect from config.
        top_k: Number of results to retrieve per query.
        output_file: Optional path to save retrieval results JSON.
        split: Dataset split to evaluate on (default: "test").
        skill_ids_filter: Optional subset of skill IDs to evaluate on.
        trust_remote_code: If None, uses env ``SKILLRET_TRUST_REMOTE_CODE``.
        embedding_cache_dir: Where to store FAISS indexes.
        use_embedding_cache: If True, load or save FAISS index.
        force_rebuild_embedding_cache: If True, always re-encode corpus.

    Returns:
        Dict mapping "test" -> metric dict.
    """
    if batch_size <= 0:
        batch_size = get_batch_size(model_path, "embed")
        print(f"Auto-detected embedding batch_size={batch_size} for {model_path}")

    bk = _behavior_key(model_path)
    model = RetModel(model_path, trust_remote_code=trust_remote_code, behavior_key=bk)

    cache_root = Path(embedding_cache_dir).expanduser().resolve() if embedding_cache_dir else EMBEDDING_CACHE_DIR
    cache_name = _embedding_cache_name(model_path)
    faiss_cache_path, meta_cache_path = _embedding_cache_paths(cache_root, cache_name)

    skills = load_corpus(split=split)
    print(f"Loaded {len(skills)} skills from corpus ({split} split)")

    index = None
    if use_embedding_cache and not force_rebuild_embedding_cache:
        index = _try_load_embedding_index(
            faiss_index_path=faiss_cache_path,
            meta_path=meta_cache_path,
            model_ref=model_path,
            n_skills=len(skills),
        )
        if index is not None:
            print(f"Loaded cached FAISS index ({index.ntotal} vectors) from {faiss_cache_path}")

    if index is None:
        skill_embeddings = model.encode_corpus(skills, batch_size)
        dim = int(skill_embeddings.shape[1])
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(skill_embeddings)
        if use_embedding_cache:
            _save_embedding_index(
                faiss_index_path=faiss_cache_path,
                meta_path=meta_cache_path,
                model_ref=model_path,
                n_skills=len(skills),
                dim=dim,
                index=index,
            )
            print(f"Saved FAISS index to {faiss_cache_path}")

    if skill_ids_filter:
        keep_positions = [i for i, s in enumerate(skills) if s["id"] in skill_ids_filter]
        skills = [skills[i] for i in keep_positions]
        vecs = np.vstack([index.reconstruct(i) for i in keep_positions]).astype(np.float32)
        dim = int(vecs.shape[1])
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(vecs)
        print(f"Filtered to {len(skills)} skills ({index.ntotal} vectors)")

    queries = load_queries(split=split)
    query_embeddings = model.encode_queries(queries, batch_size)
    distances, indices = index.search(query_embeddings, top_k)

    results: Dict[str, Dict[str, float]] = {}
    for item, ranks, dists in zip(queries, indices, distances):
        results[item["id"]] = {
            str(skills[int(r)]["id"]): float(d)
            for r, d in zip(ranks, dists)
            if r >= 0
        }

    qrels: Dict[str, Dict[str, int]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {
            str(x["id"]): int(x["relevance"]) for x in labels
        }

    metrics = trec_eval(qrels=qrels, results=results)
    collection = {split: metrics}

    if output_file:
        write_json({"metrics": collection, "retrieval": {split: results}}, output_file)

    return collection


# ---------------------------------------------------------------------------
# BM25 baseline
# ---------------------------------------------------------------------------

def eval_bm25(
    top_k: int = RERANK_TOP_K,
    output_file: str | None = None,
    split: str = "test",
) -> Dict[str, Dict[str, float]]:
    """Evaluate BM25 (lexical) retrieval on skill retrieval.

    Uses ``bm25s`` for tokenisation, indexing and search.  No GPU required.

    Returns:
        Dict mapping split -> metric dict.
    """
    import bm25s

    skills = load_corpus(split=split)
    print(f"Loaded {len(skills)} skills from corpus ({split} split)")

    doc_texts = [_embedding_text_for_skill(s) for s in skills]
    doc_ids = [s["id"] for s in skills]

    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print(f"BM25 index built ({len(doc_texts)} documents)")

    queries = load_queries(split=split)

    results: Dict[str, Dict[str, float]] = {}
    for item in tqdm(queries, desc="BM25"):
        query_text = item["query"]
        query_tokens = bm25s.tokenize(query_text, stopwords="en")
        if len(query_tokens.vocab) == 0:
            query_tokens = bm25s.tokenize("NONE", stopwords=[])

        hits, scores = retriever.retrieve(
            query_tokens, corpus=doc_ids, k=min(top_k, len(doc_ids)),
        )
        results[item["id"]] = {
            str(hits[0, i]): float(scores[0, i])
            for i in range(len(hits[0]))
            if scores[0, i] > 0
        }

    qrels: Dict[str, Dict[str, int]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {
            str(x["id"]): int(x["relevance"]) for x in labels
        }

    metrics = trec_eval(qrels=qrels, results=results)
    collection = {split: metrics}

    if output_file:
        write_json({"metrics": collection, "retrieval": {split: results}}, output_file)

    return collection


# ---------------------------------------------------------------------------
# Reranker models
# ---------------------------------------------------------------------------

def _require_rerank_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Reranking requires CUDA.")
    return torch.device("cuda", torch.cuda.current_device())


class RankModel:
    """Base class for reranking models (cross-encoders)."""

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        raise NotImplementedError

    def compute_rank_score(self, query: str, docs: list[str]) -> list[float]:
        raise NotImplementedError


class FlagRankModel(RankModel):
    """BGE rerankers via FlagEmbedding."""

    def __init__(self, model_name: str, device: torch.device, batch_size: int = DEFAULT_RERANK_BATCH_SIZE):
        self._flag_batch_size = max(1, int(batch_size))
        super().__init__(model_name, device)

    def _load_model(self, model_name: str):
        try:
            from FlagEmbedding import FlagLLMReranker, FlagReranker
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required for BAAI bge-reranker models. "
                'Install with: pip install "FlagEmbedding>=1.3.3"'
            ) from e

        _dev = str(self.device)
        if "gemma" in model_name.lower():
            reranker = FlagLLMReranker(
                model_name, use_fp16=True, batch_size=self._flag_batch_size,
                trust_remote_code=True, devices=_dev,
            )
        else:
            reranker = FlagReranker(
                model_name, use_fp16=True, batch_size=self._flag_batch_size,
                trust_remote_code=True, devices=_dev,
            )

        # XLM-RoBERTa (bge-reranker-v2-m3 etc.) has type_vocab_size=1;
        # transformers 5.x prepare_for_model emits 0/1 segment ids causing OOB.
        tvs = getattr(getattr(reranker, "model", None), "config", None)
        tvs = getattr(tvs, "type_vocab_size", None)
        if tvs is not None and tvs <= 1 and hasattr(reranker, "tokenizer"):
            import functools
            _orig_pfm = reranker.tokenizer.prepare_for_model

            @functools.wraps(_orig_pfm)
            def _pfm_zero_ttype(*args, **kwargs):
                out = _orig_pfm(*args, **kwargs)
                if "token_type_ids" in out:
                    out["token_type_ids"] = [0] * len(out["token_type_ids"])
                return out

            reranker.tokenizer.prepare_for_model = _pfm_zero_ttype

        return reranker

    def compute_rank_score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in docs]
        scores = self.model.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]


class HFRankModel(RankModel):
    """Generic rerankers via the ``rerankers`` library (T5, cross-encoder, etc.)."""

    def __init__(self, model_name: str, device: torch.device, batch_size: int = 64):
        self.batch_size = batch_size
        super().__init__(model_name, device)

    def _load_model(self, model_name: str):
        from rerankers import Reranker

        if "t5" in model_name.lower():
            r = Reranker(model_name=model_name, model_type="t5")
        else:
            model_kwargs = {"trust_remote_code": True}
            if "jina" not in model_name.lower():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            if "gte" in model_name.lower():
                model_kwargs["unpad_inputs"] = True
            r = Reranker(
                model_name=model_name,
                model_type="cross-encoder",
                dtype="bfloat16",
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"trust_remote_code": True},
            )
        hf_model = getattr(r, "model", None)
        if hf_model is not None:
            fix_rotary_embeddings(hf_model)
            hf_model.to(self.device)
            for name, buf in hf_model.named_buffers():
                if "position_ids" in name and buf.numel() > 0:
                    expected = torch.arange(buf.numel(), device=buf.device, dtype=buf.dtype)
                    if not torch.equal(buf, expected):
                        buf.copy_(expected)
        return r

    def compute_rank_score(self, query: str, docs: list[str]) -> list[float]:
        inner = self.model
        inputs = [(query, d) for d in docs]
        bs = getattr(inner, "batch_size", 64) or 64
        all_scores: list[float] = []
        dev = self.device
        with torch.inference_mode():
            for i in range(0, len(inputs), bs):
                batch = inputs[i : i + bs]
                tok = inner.tokenize(batch)
                if isinstance(tok, dict):
                    tok = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in tok.items()}
                logits = inner.model(**tok).logits.squeeze(-1)
                if inner.dtype != torch.float32:
                    logits = logits.float()
                logits = logits.detach().cpu()
                if logits.dim() == 0:
                    all_scores.append(logits.item())
                elif logits.dim() == 1:
                    all_scores.extend(logits.tolist())
                else:
                    all_scores.extend(logits[:, -1].tolist())
        return all_scores

    def compute_rank_score_multi(self, pairs: list[tuple[str, str]], batch_size: int = 1024) -> list[float]:
        """Score many (query, doc) pairs in large GPU batches."""
        inner = self.model
        dev = self.device
        all_scores: list[float] = []
        n_batches = (len(pairs) + batch_size - 1) // batch_size
        with torch.inference_mode():
            for i in tqdm(range(0, len(pairs), batch_size), total=n_batches, desc="    HF scoring"):
                batch = pairs[i : i + batch_size]
                tok = inner.tokenize(batch)
                if isinstance(tok, dict):
                    tok = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in tok.items()}
                logits = inner.model(**tok).logits.squeeze(-1)
                if inner.dtype != torch.float32:
                    logits = logits.float()
                logits = logits.detach().cpu()
                if logits.dim() == 0:
                    all_scores.append(logits.item())
                elif logits.dim() == 1:
                    all_scores.extend(logits.tolist())
                else:
                    all_scores.extend(logits[:, -1].tolist())
        return all_scores


class Qwen3RankModel(RankModel):
    """Qwen3-Reranker via transformers (yes/no logit scoring with chat template)."""

    _INSTRUCTION = SKILL_RERANK_INSTRUCTION
    _SYSTEM = (
        "Judge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    def __init__(self, model_name: str, device: torch.device, batch_size: int = DEFAULT_RERANK_BATCH_SIZE):
        self.batch_size = batch_size
        super().__init__(model_name, device)

    def _load_model(self, model_name: str):
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        model = model.to(self.device).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = get_max_seq_length(model_name, mode="rerank") or 32768

        prefix = (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        return model

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self._INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _score_batch(self, formatted_pairs: list[str]) -> list[float]:
        inputs = self.tokenizer(
            formatted_pairs, padding=False, truncation=True,
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for j, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][j] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        base_out = self.model.model(**inputs)
        last_hidden = base_out.last_hidden_state[:, -1:, :]
        logits = self.model.lm_head(last_hidden).squeeze(1)
        true_v = logits[:, self.token_true_id]
        false_v = logits[:, self.token_false_id]
        stacked = torch.stack([false_v, true_v], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return probs[:, 1].exp().tolist()

    @torch.no_grad()
    def compute_rank_score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [self._format_pair(query, d) for d in docs]
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            all_scores.extend(self._score_batch(pairs[i : i + self.batch_size]))
        return all_scores

    @torch.no_grad()
    def compute_rank_score_multi(self, pairs: list[tuple[str, str]], batch_size: int = 0) -> list[float]:
        """Score many (query, doc) pairs in large GPU batches."""
        bs = batch_size if batch_size > 0 else self.batch_size
        formatted = [self._format_pair(q, d) for q, d in pairs]
        all_scores: list[float] = []
        n_batches = (len(formatted) + bs - 1) // bs
        for i in tqdm(range(0, len(formatted), bs), total=n_batches, desc="    Qwen scoring"):
            all_scores.extend(self._score_batch(formatted[i : i + bs]))
        return all_scores


class SkillRouterRankModel(Qwen3RankModel):
    """SkillRouter-Reranker with its own instruction and prompt format."""

    _INSTRUCTION = (
        "Given a task description, judge whether the skill document "
        "is relevant and useful for completing the task"
    )

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self._INSTRUCTION}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {doc}"
        )


def _load_reranker(
    model_name: str,
    device: torch.device,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
) -> RankModel:
    key = model_name.lower()
    if "skillrouter-reranker" in key:
        return SkillRouterRankModel(model_name, device, batch_size=rerank_batch_size)
    if "skillret-reranker" in key or "qwen3-reranker" in key:
        return Qwen3RankModel(model_name, device, batch_size=rerank_batch_size)
    if "bge-reranker" in key:
        return FlagRankModel(model_name, device, batch_size=rerank_batch_size)
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        arch = (cfg.get("architectures") or [""])[0].lower()
        if "qwen3" in arch:
            return Qwen3RankModel(model_name, device, batch_size=rerank_batch_size)
    return HFRankModel(model_name, device, batch_size=rerank_batch_size)


def _rerank_skill_text(skill: dict) -> str:
    """Build document text for reranking (full text)."""
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    body = (skill.get("skill_md") or "").strip()
    return f"{name} | {desc} | {body}"


def eval_rerank(
    reranker_model: str,
    first_stage_file: str,
    from_top_k: int = RERANK_TOP_K,
    output_file: str | None = None,
    rerank_batch_size: int = 0,
    split: str = "test",
) -> Dict[str, Dict[str, float]]:
    """Rerank first-stage retrieval results with a cross-encoder.

    Args:
        reranker_model: HuggingFace model id or local path to a reranker checkpoint.
        first_stage_file: JSON output from ``eval_retrieval`` containing per-split
            ``{query_id: {skill_id: score}}`` candidates.
        from_top_k: How many first-stage candidates to rerank per query.
        output_file: Optional path to save reranked results.
        rerank_batch_size: Batch size for reranker inference. 0 = auto-detect from config.
        split: Dataset split (default: "test").

    Returns:
        Dict mapping split -> metric dict.
    """
    device = _require_rerank_device()
    print(f"Rerank device: {device}")
    if rerank_batch_size <= 0:
        rerank_batch_size = get_batch_size(reranker_model, "rerank")
        print(f"Auto-detected rerank batch_size={rerank_batch_size} for {reranker_model}")
    print(f"Loading reranker: {reranker_model} (batch_size={rerank_batch_size})")
    model = _load_reranker(reranker_model, device, rerank_batch_size=rerank_batch_size)

    skills = load_corpus(split=split)
    skill_map = {str(s["id"]): s for s in skills}
    print(f"Loaded {len(skills)} skills from corpus ({split} split)")

    first_stage_raw = load_json(first_stage_file)
    first_stage = first_stage_raw.get("retrieval", first_stage_raw)
    print(f"Loaded first-stage results from {first_stage_file}")

    # Find the retrieval results (may be keyed by split name or be flat)
    if split in first_stage:
        task_results = first_stage[split]
    else:
        task_results = next(iter(first_stage.values())) if first_stage else {}

    queries = load_queries(split=split)
    use_multi = hasattr(model, "compute_rank_score_multi")
    result: Dict[str, Dict[str, float]] = {}

    if use_multi:
        all_pairs: list[tuple[str, str]] = []
        pair_map: list[tuple[str, list[dict], int, int]] = []
        for item in queries:
            qid = item["id"]
            candidates_scores = task_results.get(qid, {})
            sorted_candidates = sorted(
                candidates_scores.items(), key=lambda x: x[1], reverse=True
            )[:from_top_k]
            candidate_skills = [
                skill_map[sid] for sid, _ in sorted_candidates if sid in skill_map
            ]
            if not candidate_skills:
                result[qid] = {}
                continue
            doc_texts = [_rerank_skill_text(s) for s in candidate_skills]
            start = len(all_pairs)
            all_pairs.extend((item["query"], d) for d in doc_texts)
            pair_map.append((qid, candidate_skills, start, len(all_pairs)))

        bs = getattr(model, "batch_size", 64) or 64
        sort_idx = sorted(range(len(all_pairs)),
                          key=lambda i: len(all_pairs[i][0]) + len(all_pairs[i][1]))
        sorted_pairs = [all_pairs[i] for i in sort_idx]
        print(f"  Cross-query batching: {len(all_pairs)} pairs, batch_size={bs} (length-sorted)")
        sorted_scores = model.compute_rank_score_multi(sorted_pairs, batch_size=bs)
        all_scores = [0.0] * len(sorted_scores)
        for orig_i, sc in zip(sort_idx, sorted_scores):
            all_scores[orig_i] = sc

        for qid, candidate_skills, start, end in pair_map:
            result[qid] = {
                str(s["id"]): float(sc)
                for s, sc in zip(candidate_skills, all_scores[start:end])
            }
    else:
        for item in tqdm(queries, desc="Reranking", leave=False):
            qid = item["id"]
            candidates_scores = task_results.get(qid, {})
            sorted_candidates = sorted(
                candidates_scores.items(), key=lambda x: x[1], reverse=True
            )[:from_top_k]
            candidate_skills = [
                skill_map[sid] for sid, _ in sorted_candidates if sid in skill_map
            ]
            if not candidate_skills:
                result[qid] = {}
                continue
            doc_texts = [_rerank_skill_text(s) for s in candidate_skills]
            scores = model.compute_rank_score(item["query"], doc_texts)
            result[qid] = {
                str(s["id"]): float(sc)
                for s, sc in zip(candidate_skills, scores)
            }

    qrels: Dict[str, Dict[str, int]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {
            str(x["id"]): int(x["relevance"]) for x in labels
        }

    metrics = trec_eval(qrels=qrels, results=result)
    collection = {split: metrics}

    if output_file:
        write_json({"metrics": collection, "retrieval": {split: result}}, output_file)

    return collection


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(
    results: Dict[str, Dict[str, float]],
    metrics: list[str] | None = None,
) -> None:
    if metrics is None:
        metrics = ["NDCG@5", "NDCG@10", "NDCG@15", "Recall@5", "Recall@10", "Recall@15"]

    table_data = []
    avg = {m: 0.0 for m in metrics}
    for task, scores in results.items():
        row = [task] + [f"{scores.get(m, 0) * 100:.2f}" for m in metrics]
        table_data.append(row)
        for m in metrics:
            avg[m] += scores.get(m, 0)

    n = max(len(results), 1)
    table_data.append(["Average"] + [f"{v / n * 100:.2f}" for v in avg.values()])

    headers = ["Task"] + metrics
    try:
        from tabulate import tabulate
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        widths = [max(len(str(c)) for c in col) for col in zip(headers, *table_data)]
        header = " | ".join(f"{h:^{w}}" for h, w in zip(headers, widths))
        print(f"| {header} |")
        print("-+-".join("-" * w for w in widths))
        for row in table_data:
            print("| " + " | ".join(f"{c:^{w}}" for c, w in zip(row, widths)) + " |")
