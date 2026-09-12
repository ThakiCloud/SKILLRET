# SkillRet: A Benchmark for AI Agent Skill Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2605.05726-b31b1b.svg)](https://arxiv.org/abs/2605.05726)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ThakiCloud%2FSKILLRET-yellow)](https://huggingface.co/datasets/ThakiCloud/SKILLRET)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository is the official implementation of **SkillRet: A Benchmark for AI Agent Skill Retrieval**.

📄 **Paper**: [SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arXiv:2605.05726)](https://arxiv.org/abs/2605.05726)

Given a natural-language user query (e.g., *"Can you review my staged changes before I commit?"*), the task is to retrieve the most relevant skill(s) from a library of 6,006 AI agent skills collected from open-source repositories.

## Requirements

- Python 3.13
- CUDA 12.8+ (for GPU-accelerated FAISS and flash-attn)
- 1+ NVIDIA GPU (evaluation); 4+ GPUs (training)

To install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Adjust FLASH_ATTN_CUDA_ARCHS in pyproject.toml to match your GPU:
#   SM 100 (B200), SM 90 (H100), SM 80 (A100)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

uv sync
source .venv/bin/activate
```

Set up environment:

```bash
# Create .env for HuggingFace cache and tokens
cat > .env << 'EOF'
HF_HOME=/path/to/your/hf_cache     # HuggingFace cache directory
HF_TOKEN=hf_your_token_here         # For gated models (optional)
EOF
```

All models are loaded by their HuggingFace ID and **downloaded automatically** on first run. Cached under `$HF_HOME`.

## Dataset

The benchmark dataset is hosted on HuggingFace:
[ThakiCloud/SKILLRET](https://huggingface.co/datasets/ThakiCloud/SKILLRET)

> **Dataset version note.** The test split was revised after the paper's original
> evaluation. The published dataset now contains 6,006 test skills, 4,392 test queries
> and 7,187 test labels; the original evaluation used 6,660 / 4,997 / 8,347. Benchmark
> scores further down this page were produced on that **original** split and are not
> convertible to the current one — do not compare the two directly. The train split is
> unchanged.
>
> The Hub head is mutable, so an unpinned run is not reproducible. Pin the revision you
> evaluate on (see [Loading the data](#loading-the-data)). The study *Where
> Post-Training Quantization Breaks Text Embedders* pins revision `a050ad2`, which is
> the current head at the time of writing.

| Subset  | Split | Records | Description                           |
|---------|-------|--------:|---------------------------------------|
| skills  | test  |   6,006 | Evaluation skill corpus               |
| queries | test  |   4,392 | Evaluation queries (Claude Opus 4.6)  |
| qrels   | test  |   7,187 | Binary relevance labels               |
| skills  | train |  10,123 | Training skill corpus                 |
| queries | train |  63,259 | Training queries (Qwen3.5-122B-A10B)  |
| qrels   | train | 127,190 | Training relevance labels             |

### Loading the data

The evaluation code **automatically downloads** the dataset from HuggingFace on first run. HuggingFace's `datasets` library handles caching in `~/.cache/huggingface/`. No manual download step is needed.

```python
# The evaluation functions load data automatically:
from skillret.eval import eval_retrieval, print_results
results = eval_retrieval(model_path="Qwen/Qwen3-Embedding-8B")

# Or load data directly:
from skillret.eval import load_corpus, load_queries
skills = load_corpus()    # 6,006 skills (test split)
queries = load_queries()  # 4,392 queries (test split)

# Pin the Hub revision so the run is reproducible:
skills = load_corpus(revision="a050ad2")
queries = load_queries(revision="a050ad2")
```

Every entry point also honours the `SKILLRET_DATASET_REVISION` environment variable, which
pins the revision without touching call sites:

```bash
SKILLRET_DATASET_REVISION=a050ad2 bash scripts/run_eval_embedding.sh
```

Unset, it resolves to the Hub head, which is the previous behaviour.

## Evaluation

### Embedding retrieval (first stage)

```bash
# Evaluate all models across multiple GPUs
NUM_GPUS=8 bash scripts/run_eval_embedding.sh

# Single model evaluation (auto-downloads from HuggingFace)
python -c "
from skillret.eval import eval_retrieval, print_results
results = eval_retrieval(
    model_path='Qwen/Qwen3-Embedding-8B',
    top_k=20,
    output_file='results/embed/qwen3-8b.json',
)
print_results(results)
"
```

### Reranking (second stage)

```bash
# Evaluate all rerankers on first-stage results
NUM_GPUS=8 bash scripts/run_eval_rerank.sh
```

## Training

All training scripts load data from the HuggingFace dataset automatically (no manual download needed). Models are specified by their HuggingFace IDs and downloaded on first use.

### Run all training (sequential)

```bash
nohup bash train/run_all.sh > train/run_all.log 2>&1 &
```

This runs all three phases sequentially with GPU cleanup between runs:
1. SkillRet-Embedding-0.6B (~5h on 4x GPU)
2. SkillRet-Embedding-8B (~16h on 4x GPU)
3. SkillRet-Reranker-0.6B (~6h on 8x GPU, including hard negative mining from SkillRet-Embedding-0.6B)

### Embedding model fine-tuning

Fine-tune embedding models with in-batch Multiple Negatives Ranking Loss:

```bash
# SkillRet-Embedding-0.6B (4 GPU DDP, effective_batch=384)
torchrun --nproc_per_node=4 train/4gpu-qwen3-0.6b/train.py

# SkillRet-Embedding-8B (4 GPU DDP, effective_batch=80)
torchrun --nproc_per_node=4 train/4gpu-qwen3-8b/train.py
```

| Script | Base Model | Output | Effective Batch |
|--------|-----------|--------|-----------------|
| `train/4gpu-qwen3-0.6b/train.py` | Qwen/Qwen3-Embedding-0.6B | SkillRet-Embedding-0.6B | 384 |
| `train/4gpu-qwen3-8b/train.py` | Qwen/Qwen3-Embedding-8B | SkillRet-Embedding-8B | 80 |

### Reranker fine-tuning

Fine-tune Qwen3-Reranker-0.6B with BCE (yes/no SFT) loss using hard negatives mined from SkillRet-Embedding-0.6B:

#### Step 1: Mine hard negatives

Retrieves the top-100 non-GT candidates per training query and saves the ranked list. Only needs to be done once.

```bash
CUDA_VISIBLE_DEVICES=0 python train/reranker-ft/train.py --mine-hard-negatives \
    --config train/reranker-ft/configs/qwen3-reranker-0.6b-sft-emb06b-best.yaml
# -> saves data/hard_negatives_emb06b.json
```

#### Step 2: Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    train/reranker-ft/train.py \
    --config train/reranker-ft/configs/qwen3-reranker-0.6b-sft-emb06b-best.yaml
```

Key training settings: BCE loss, per-positive grouping, 1 epoch, ~6h on 8× B200.

## Pre-trained Models

| Model | Type | HuggingFace |
|-------|------|-------------|
| SKILLRET-Embedding-0.6B | Embedding | [ThakiCloud/SKILLRET-Embedding-0.6B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B) |
| SKILLRET-Embedding-8B | Embedding | [ThakiCloud/SKILLRET-Embedding-8B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-8B) |
| SKILLRET-Reranker-0.6B | Reranker | ThakiCloud/SKILLRET-Reranker-0.6B |

## Results

> Produced on the original 6,660-skill / 4,997-query test split, not on the split the
> dataset publishes today. See the [dataset version note](#dataset).

### Embedding Retrieval

| Model | Params | NDCG@5 | NDCG@10 | Recall@10 | Comp.@10 |
|-------|--------|--------|---------|-----------|----------|
| BM25 | -- | 46.47 | 48.86 | 56.55 | 41.09 |
| bge-small-en-v1.5 | 33M | 49.62 | 51.68 | 57.47 | 40.72 |
| snowflake-arctic-embed-s | 33M | 51.01 | 52.99 | 58.84 | 42.48 |
| e5-small-v2 | 118M | 39.83 | 41.82 | 48.65 | 34.02 |
| e5-large-v2 | 335M | 48.01 | 50.21 | 57.42 | 41.95 |
| bge-large-en-v1.5 | 335M | 53.75 | 55.82 | 61.40 | 44.63 |
| F2LLM-v2-80M | 80M | 43.32 | 45.52 | 52.24 | 36.66 |
| harrier-oss-v1-270m | 270M | 58.83 | 61.17 | 67.61 | 51.11 |
| pplx-embed-v1-0.6b | 0.6B | 48.11 | 50.72 | 60.03 | 44.07 |
| Qwen3-Embedding-0.6B | 0.6B | 56.23 | 58.35 | 64.89 | 47.27 |
| jina-embeddings-v5-text-small | 0.6B | 57.21 | 59.50 | 65.77 | 49.03 |
| harrier-oss-v1-0.6b | 0.6B | 64.24 | 66.55 | 73.09 | 57.37 |
| NV-Embed-v1 | 7B | 50.71 | 53.12 | 60.96 | 43.87 |
| Qwen3-Embedding-8B | 8B | 57.57 | 59.98 | 67.06 | 50.01 |
| Octen-Embedding-8B | 8B | 60.35 | 62.56 | 68.17 | 51.31 |
| KaLM-Gemma3-12B | 12B | 52.68 | 55.38 | 63.94 | 47.85 |
| **[SKILLRET-Embedding-0.6B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B) (ours)** | 0.6B | 75.57 | 78.03 | 85.42 | 75.09 |
| **[SKILLRET-Embedding-8B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-8B) (ours)** | 8B | **81.23** | **83.45** | **91.23** | **84.63** |

### Reranking (SkillRet-Embedding-0.6B top-20)

| Reranker | NDCG@5 | NDCG@10 | Recall@10 | Comp.@10 |
|----------|--------|---------|-----------|----------|
| *Embed only* | 75.57 | 78.03 | 85.42 | 75.09 |
| jina-reranker-v2 | 69.92 | 73.14 | 83.93 | 73.06 |
| Qwen3-Reranker-0.6B | 72.84 | 75.81 | 85.86 | 75.48 |
| Qwen3-Reranker-4B | 73.24 | 76.20 | 86.18 | 76.09 |
| Qwen3-Reranker-8B | 73.21 | 76.09 | 86.48 | 76.53 |
| **SkillRet-Reranker-0.6B (ours)** | **80.71** | **82.18** | **87.61** | **78.95** |

## Repository Structure

```
skillret-benchmark/
├── skillret/                    # Evaluation package
│   ├── config.py                # Model configs, batch sizes, prompts
│   ├── eval.py                  # Retrieval + reranking evaluation
│   ├── _compat.py               # Transformers 5.x compatibility patches
│   └── utils.py                 # I/O helpers
├── train/                       # Fine-tuning code
│   ├── 4gpu-qwen3-0.6b/        # SkillRet-Embedding-0.6B training
│   ├── 4gpu-qwen3-8b/          # SkillRet-Embedding-8B training
│   ├── reranker-ft/             # SkillRet-Reranker-0.6B SFT training
│   │   └── configs/             # Training YAML configs
│   └── run_all.sh               # Run all training sequentially
├── scripts/                     # Evaluation orchestration
│   ├── run_eval_embedding.sh    # Multi-GPU embedding eval
│   └── run_eval_rerank.sh       # Multi-GPU reranking eval
└── pyproject.toml               # Dependencies
```

## Metrics

All metrics computed via `pytrec_eval` at k={5, 10, 15}:

| Metric | Description |
|--------|-------------|
| NDCG@k | Normalized Discounted Cumulative Gain |
| Recall@k | Fraction of relevant skills retrieved |
| Completeness@k | Fraction of queries with perfect recall |
| MAP@k | Mean Average Precision |

## Citation

If you use SkillRet, please cite:

```bibtex
@article{cho2026skillret,
  title   = {SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents},
  author  = {Cho, Hongcheol and Kang, Ryangkyung and Kim, Youngeun},
  journal = {arXiv preprint arXiv:2605.05726},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.05726}
}
```

**Paper**: [https://arxiv.org/abs/2605.05726](https://arxiv.org/abs/2605.05726)

## License

Apache 2.0
