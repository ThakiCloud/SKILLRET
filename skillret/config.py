import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Per-model FAISS index + metadata. Override with SKILLRET_INDEX_DIR so that
# indexes built for different corpora do not overwrite each other.
EMBEDDING_CACHE_DIR = Path(os.getenv("SKILLRET_INDEX_DIR") or (DATA_DIR / "indexes"))

HF_DATASET_ID = "ThakiCloud/SKILLRET"

# Local benchmark directory holding {skills,queries,qrels}/{split}.jsonl. When set,
# it takes precedence over the HuggingFace dataset above.
LOCAL_DATA_DIR = Path(os.getenv("SKILLRET_DATA_DIR")) if os.getenv("SKILLRET_DATA_DIR") else None


SKILL_QUERY_PROMPT = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "

SKILL_RERANK_INSTRUCTION = (
    "Given a skill search query, judge whether the skill document "
    "is relevant and useful for the query"
)

# ---------------------------------------------------------------------------
# Unified embedding model registry.
#
# Each entry may contain:
#   batch_size     – encoding batch size (stress-tested on single GPU, 191 GB VRAM)
#   query_prefix / doc_prefix  – prepended to raw text before encoding
#   query_kwargs / doc_kwargs  – passed to model.encode() for queries / docs
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_CONFIG: dict[str, dict] = {
    # --- Lexical Baseline ---
    "BM25": {},
    # --- Encoder-only (Bidirectional) ---
    "BAAI/bge-small-en-v1.5": {
        "batch_size": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "intfloat/e5-small-v2": {
        "batch_size": 1024,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    "Snowflake/snowflake-arctic-embed-s": {
        "batch_size": 1024,
        "query_kwargs": {"prompt_name": "query"},
    },
    "BAAI/bge-large-en-v1.5": {
        "batch_size": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "intfloat/e5-large-v2": {
        "batch_size": 1024,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    # --- Decoder-only (Causal LM) ---
    "codefuse-ai/F2LLM-v2-80M": {
        "batch_size": 16,
        "query_kwargs": {"prompt_name": "query"},
    },
    "microsoft/harrier-oss-v1-270m": {
        "batch_size": 32,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "perplexity-ai/pplx-embed-v1-0.6b": {
        "batch_size": 16,
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "batch_size": 16,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "microsoft/harrier-oss-v1-0.6b": {
        "batch_size": 32,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "jinaai/jina-embeddings-v5-text-small": {
        "batch_size": 16,
        "query_kwargs": {"task": "retrieval", "prompt_name": "query"},
        "doc_kwargs": {"task": "retrieval", "prompt_name": "document"},
    },
    "nvidia/NV-Embed-v1": {
        "batch_size": 4,
        "max_seq_length": 32768,
    },
    "Octen/Octen-Embedding-8B": {
        "batch_size": 8,
        "max_seq_length": 32768,
        "doc_prefix": "- ",
        "query_kwargs": {"prompt_name": None},
        "doc_kwargs": {"prompt_name": None},
    },
    "Qwen/Qwen3-Embedding-8B": {
        "batch_size": 8,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "tencent/KaLM-Embedding-Gemma3-12B-2511": {
        "batch_size": 4,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": "Instruct: Given a query, retrieve documents that answer the query \nQuery: "},
    },
    # --- Fine-tuned models ---
    "ThakiCloud/SKILLRET-Embedding-8B": {
        "batch_size": 8,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "ThakiCloud/SKILLRET-Embedding-0.6B": {
        "batch_size": 16,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": SKILL_QUERY_PROMPT},
    },
    "pipizhao/SkillRouter-Embedding-0.6B": {
        "batch_size": 16,
        "max_seq_length": 32768,
        "query_kwargs": {"prompt": "Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:"},
    },
}

EMBEDDING_MODELS = list(EMBEDDING_MODEL_CONFIG.keys())

# ---------------------------------------------------------------------------
# Unified reranking model registry.
#
# Each entry may contain:
#   batch_size  – reranking batch size (stress-tested on single GPU, 191 GB VRAM)
# ---------------------------------------------------------------------------
RERANKING_MODEL_CONFIG: dict[str, dict] = {
    # --- Encoder-only (Cross-encoder) ---
    "jinaai/jina-reranker-v2-base-multilingual": {"batch_size": 1024},
    # --- Decoder-only (LLM-based) ---
    "Qwen/Qwen3-Reranker-0.6B":               {"batch_size": 32, "max_seq_length": 32768},
    "Qwen/Qwen3-Reranker-4B":                  {"batch_size": 4, "max_seq_length": 32768},
    "Qwen/Qwen3-Reranker-8B":                  {"batch_size": 4, "max_seq_length": 32768},
    "ThakiCloud/SKILLRET-Reranker-0.6B":        {"batch_size": 32, "max_seq_length": 32768},
    "pipizhao/SkillRouter-Reranker-0.6B":  {"batch_size": 32, "max_seq_length": 32768},
}

RERANKING_MODELS = list(RERANKING_MODEL_CONFIG.keys())

RERANK_TOP_K = 20

DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_RERANK_BATCH_SIZE = 16


def get_st_config(behavior_key: str) -> dict:
    """Look up ST encode config by matching behavior_key against the model registry."""
    for key, cfg in EMBEDDING_MODEL_CONFIG.items():
        if key in behavior_key:
            return {
                k: v for k, v in cfg.items()
                if k in ("query_prefix", "doc_prefix", "query_kwargs", "doc_kwargs")
            }
    return {}


def get_batch_size(model_path: str, mode: str = "embed") -> int:
    """Look up batch size by matching the model path against known model short names."""
    default = DEFAULT_EMBED_BATCH_SIZE if mode == "embed" else DEFAULT_RERANK_BATCH_SIZE
    registry = EMBEDDING_MODEL_CONFIG if mode == "embed" else RERANKING_MODEL_CONFIG
    cfg = _best_match(registry, model_path)
    return cfg.get("batch_size", default) if cfg else default


def _best_match(registry: dict[str, dict], model_path: str) -> dict | None:
    """Return the config entry whose key is the longest substring match in model_path."""
    best_key, best_cfg = None, None
    for key, cfg in registry.items():
        if key in model_path:
            if best_key is None or len(key) > len(best_key):
                best_key, best_cfg = key, cfg
    return best_cfg


def get_max_seq_length(model_path: str, mode: str = "embed") -> int | None:
    """Look up max_seq_length by matching the model path against known model short names.

    Returns None if not explicitly configured (model uses its own default).
    """
    registry = EMBEDDING_MODEL_CONFIG if mode == "embed" else RERANKING_MODEL_CONFIG
    cfg = _best_match(registry, model_path)
    return cfg.get("max_seq_length") if cfg else None


