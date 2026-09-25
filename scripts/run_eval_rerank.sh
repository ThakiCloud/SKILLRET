#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Worker-pool multi-GPU reranker evaluator
#
# Reranks first-stage retrieval results from multiple embedding models
# using all reranking models in RERANKING_MODELS (config.py).
# Models are loaded by HuggingFace ID and cached under $HF_HOME.
#
# Usage:
#   NUM_GPUS=7 bash scripts/run_eval_rerank.sh
#   MODELS_FILTER="Qwen3" bash scripts/run_eval_rerank.sh
#   FIRST_STAGE_FILTER="SkillRet" bash scripts/run_eval_rerank.sh
#   RERANK_TOP_K="10,20,50" bash scripts/run_eval_rerank.sh
#
# Environment variables:
#   NUM_GPUS            Number of GPUs (default: 7)
#   MODELS_FILTER       Comma-separated substrings to match reranker names
#   FIRST_STAGE_FILTER  Comma-separated substrings to match first-stage names
#   RERANK_TOP_K        First-stage depth, comma-separated for multi-k (default: 20)
#   OUTPUT_DIR          Directory for rerank results (default: results/rerank)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

NUM_GPUS="${NUM_GPUS:-7}"
RERANK_TOP_K="${RERANK_TOP_K:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rerank}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# First-stage embedding results to rerank
# ---------------------------------------------------------------------------
FIRST_STAGE_DIR="${FIRST_STAGE_DIR:-results/embed}"

declare -A FIRST_STAGE_FILES
for tier in \
    Snowflake_snowflake-arctic-embed-s \
    microsoft_harrier-oss-v1-270m \
    microsoft_harrier-oss-v1-0.6b \
    Qwen_Qwen3-Embedding-0.6B \
    Qwen_Qwen3-Embedding-8B \
    ThakiCloud_SKILLRET-Embedding-0.6B \
    ThakiCloud_SKILLRET-Embedding-8B ; do
    FIRST_STAGE_FILES["$tier"]="${FIRST_STAGE_DIR}/${tier}.json"
done

# Filter and validate first-stage files
VALID_FIRST_STAGES=()
for tier in "${!FIRST_STAGE_FILES[@]}"; do
    fs_file="${FIRST_STAGE_FILES[$tier]}"
    [[ ! -f "$fs_file" ]] && { echo "WARN: skipping $tier (not found: $fs_file)"; continue; }
    if [[ -n "${FIRST_STAGE_FILTER:-}" ]]; then
        match=0
        IFS=',' read -ra FILTERS <<< "$FIRST_STAGE_FILTER"
        for filt in "${FILTERS[@]}"; do
            [[ "$tier" == *"$(echo "$filt" | xargs)"* ]] && { match=1; break; }
        done
        [[ $match -eq 0 ]] && continue
    fi
    VALID_FIRST_STAGES+=("${tier}|${fs_file}")
done

[[ ${#VALID_FIRST_STAGES[@]} -eq 0 ]] && { echo "ERROR: no valid first-stage files."; exit 1; }

# ---------------------------------------------------------------------------
# Build reranker model list from config.py
# ---------------------------------------------------------------------------
mapfile -t ALL_RERANKERS < <(
    python -c "
from skillret.config import RERANKING_MODELS
for m in RERANKING_MODELS:
    print(m)
"
)

RERANKERS=()
for model_id in "${ALL_RERANKERS[@]}"; do
    if [[ -n "${MODEL_BASE_DIR:-}" ]]; then
        [[ ! -d "${MODEL_BASE_DIR}/${model_id}" ]] && { echo "WARN: skipping $model_id (not found at ${MODEL_BASE_DIR}/${model_id})"; continue; }
    fi
    if [[ -n "${MODELS_FILTER:-}" ]]; then
        match=0
        IFS=',' read -ra FILTERS <<< "$MODELS_FILTER"
        for filt in "${FILTERS[@]}"; do
            [[ "$model_id" == *"$(echo "$filt" | xargs)"* ]] && { match=1; break; }
        done
        [[ $match -eq 0 ]] && continue
    fi
    RERANKERS+=("$model_id")
done

[[ ${#RERANKERS[@]} -eq 0 ]] && { echo "ERROR: no reranker models matched."; exit 1; }

# Build job queue: "reranker_id|tier|first_stage_file|top_k"
IFS=',' read -ra TOP_K_VALUES <<< "$RERANK_TOP_K"

JOBS=()
for reranker_id in "${RERANKERS[@]}"; do
    for entry in "${VALID_FIRST_STAGES[@]}"; do
        tier="${entry%%|*}"
        fs_file="${entry#*|}"
        for tk in "${TOP_K_VALUES[@]}"; do
            JOBS+=("${reranker_id}|${tier}|${fs_file}|$(echo "$tk" | xargs)")
        done
    done
done

TOTAL=${#JOBS[@]}
echo "=========================================="
echo " Rerank evaluation: ${#RERANKERS[@]} rerankers × ${#VALID_FIRST_STAGES[@]} first-stages × ${#TOP_K_VALUES[@]} top-k = $TOTAL jobs"
echo " GPUs: $NUM_GPUS  TOP_K=${TOP_K_VALUES[*]}"
echo "=========================================="
printf '  Reranker: %s\n' "${RERANKERS[@]}"
printf '  First-stage: %s\n' "${VALID_FIRST_STAGES[@]}"
echo ""

# ---------------------------------------------------------------------------
# Queue management (flock-based)
# ---------------------------------------------------------------------------
QUEUE_DIR="$(mktemp -d)"
QUEUE_FILE="$QUEUE_DIR/queue.txt"
QUEUE_LOCK="$QUEUE_DIR/queue.lock"
DONE_FILE="$QUEUE_DIR/done.count"

printf '%s\n' "${JOBS[@]}" > "$QUEUE_FILE"
echo 0 > "$DONE_FILE"

pop_next_job() {
    (
        flock 9
        local job
        job="$(head -n1 "$QUEUE_FILE" 2>/dev/null || true)"
        if [[ -n "$job" ]]; then
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
            mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        fi
        echo "$job"
    ) 9>"$QUEUE_LOCK"
}

increment_done() {
    ( flock 9; echo $(( $(cat "$DONE_FILE") + 1 )) > "$DONE_FILE" ) 9>"$QUEUE_LOCK"
}

get_done_count() {
    ( flock 9; cat "$DONE_FILE" ) 9>"$QUEUE_LOCK"
}

# ---------------------------------------------------------------------------
# Reranking evaluation
# ---------------------------------------------------------------------------
run_rerank() {
    local gpu_id="$1" model_id="$2" tier="$3" first_stage_file="$4" top_k="$5"
    local short_name="${model_id//\//_}"
    local output_json="${OUTPUT_DIR}/rerank_${short_name}_${tier}_k${top_k}.json"
    local log="${OUTPUT_DIR}/rerank_${short_name}_${tier}_k${top_k}.log"

    if [[ -f "$output_json" ]]; then
        echo "[GPU $gpu_id] SKIP (exists): $output_json"
        return 0
    fi

    local model_ref="${MODEL_BASE_DIR:+${MODEL_BASE_DIR}/}${model_id}"
    echo "[GPU $gpu_id] Rerank: $short_name on $tier k=$top_k → $log"
    CUDA_VISIBLE_DEVICES="$gpu_id" python -c "
from skillret.eval import eval_rerank, print_results
results = eval_rerank(
    reranker_model='${model_ref}',
    first_stage_file='${first_stage_file}',
    from_top_k=${top_k},
    output_file='${output_json}',
)
print_results(results)
" 2>&1 | tee "$log"
}

# ---------------------------------------------------------------------------
# GPU worker loop
# ---------------------------------------------------------------------------
gpu_worker() {
    local gpu_id="$1"
    while true; do
        local job
        job="$(pop_next_job)"
        [[ -z "$job" ]] && break

        IFS='|' read -r model_id tier fs_file top_k <<< "$job"
        echo "[GPU $gpu_id] ▶ $model_id on $tier (k=$top_k)"

        if run_rerank "$gpu_id" "$model_id" "$tier" "$fs_file" "$top_k"; then
            echo "[GPU $gpu_id] ✓ Done: $model_id on $tier k=$top_k"
        else
            echo "[GPU $gpu_id] ✗ FAILED: $model_id on $tier k=$top_k" >&2
        fi
        increment_done
    done
}

# ---------------------------------------------------------------------------
# Launch workers and wait
# ---------------------------------------------------------------------------
MAIN_LOG="${OUTPUT_DIR}/run_${TIMESTAMP}.log"
echo "Main log: $MAIN_LOG"

WORKER_PIDS=()
for (( gpu=0; gpu < NUM_GPUS && gpu < TOTAL; gpu++ )); do
    gpu_worker "$gpu" >> "$MAIN_LOG" 2>&1 &
    WORKER_PIDS+=($!)
done
echo "Launched ${#WORKER_PIDS[@]} workers"

while true; do
    done_count="$(get_done_count)"
    still_running=0
    for pid in "${WORKER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && (( still_running++ )) || true
    done
    echo "[$(date '+%H:%M:%S')] Progress: $done_count/$TOTAL done, $still_running workers active"
    [[ $still_running -eq 0 ]] && break
    sleep 30
done

echo ""
echo "=========================================="
echo " All $TOTAL reranking evaluations complete."
echo " Results: ${OUTPUT_DIR}/*.json"
echo "=========================================="

rm -rf "$QUEUE_DIR"
