#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 11 ]; then
  echo "Usage: run_dataset.sh MODEL_KEY MANIFEST DATA_ROOT BASE_MODEL ADAPTER REVIEWER_MODEL JUDGE_MODEL OUTPUT_DIR TARGET_TORCHRUN COMMON_TORCHRUN COMMON_PYTHON" >&2
  exit 2
fi
MODEL_KEY=$1
MANIFEST=$2
DATA_ROOT=$3
BASE_MODEL=$4
ADAPTER=$5
REVIEWER_MODEL=$6
JUDGE_MODEL=$7
OUTPUT_DIR=$8
TARGET_TORCHRUN=$9
COMMON_TORCHRUN=${10}
COMMON_PYTHON=${11}
MANIFEST=$(realpath "$MANIFEST")
DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_MODEL=$(realpath "$BASE_MODEL")
ADAPTER=$(realpath "$ADAPTER")
REVIEWER_MODEL=$(realpath "$REVIEWER_MODEL")
JUDGE_MODEL=$(realpath "$JUDGE_MODEL")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"
STEM=$(basename "$MANIFEST")
STEM=${STEM%.*}
run_generate() {
  state=$1
  if [ "$state" = "orig" ]; then
    "$TARGET_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=6 scripts/mm_eval.py generate \
      --model-key "$MODEL_KEY" --state orig --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
      --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR"
  else
    "$TARGET_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=6 scripts/mm_eval.py generate \
      --model-key "$MODEL_KEY" --state m1 --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
      --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output-dir "$OUTPUT_DIR"
  fi
  "$COMMON_PYTHON" -B scripts/merge_shards.py --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
    --model-key "$MODEL_KEY" --state "$state" --num-shards 6
}
run_judge() {
  state=$1
  generation="$OUTPUT_DIR/$STEM.$state.jsonl"
  judgment="$OUTPUT_DIR/$STEM.$state.judgments.jsonl"
  "$COMMON_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=6 scripts/judge_baseline_local.py \
    --input "$generation" --model-key "$MODEL_KEY" --model-path "$JUDGE_MODEL" \
    --config configs/judge_local.json --out "$judgment" --num-shards 6
  "$COMMON_PYTHON" -B scripts/merge_judge_baseline_local.py --input "$generation" \
    --model-key "$MODEL_KEY" --config configs/judge_local.json --out "$judgment" --num-shards 6
}
run_generate orig
run_generate m1
"$COMMON_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=6 scripts/mm_eval.py review \
  --model-key "$MODEL_KEY" --manifest "$MANIFEST" --data-root "$DATA_ROOT" --m1-input-dir "$OUTPUT_DIR" \
  --reviewer-model "$REVIEWER_MODEL" --output-dir "$OUTPUT_DIR"
"$COMMON_PYTHON" -B scripts/merge_shards.py --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
  --model-key "$MODEL_KEY" --state m1_m3_v8 --num-shards 6
run_judge orig
run_judge m1
run_judge m1_m3_v8
