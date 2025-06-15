#!/bin/bash
# Run generation and evaluation for multiple profile lengths on two datasets

set -e

DATASETS=("MedCorpus" "LitSearch")
LENGTHS=(200 250 300)

BASE_DATA_DIR="/workspace/PerMed/data"
RESULTS_DIR="./results"

for DATASET in "${DATASETS[@]}"; do
  for LEN in "${LENGTHS[@]}"; do
    echo "=== Dataset: $DATASET | Profile length: ${LEN} tokens ==="

    python run.py --mode generate_narratives \
      --dataset_name "$DATASET" \
      --personalized_text_max_length "$LEN" \
      --personalized_text_target_length "$LEN" \
      --llm_model "llama3:8b"

    python run.py --mode rerank \
      --dataset_name "$DATASET" \
      --personalized_text_max_length "$LEN" \
      --personalized_text_target_length "$LEN" \
      --llm_model "llama3:8b"

    GT_PATH="${BASE_DATA_DIR}/${DATASET}/ground_truth.jsonl"
    PRED_FILE="${RESULTS_DIR}/${DATASET}/ranked_jina_profileOnly_L${LEN}_llama3-8b_top10.jsonl"
    BASELINE_FILE="${RESULTS_DIR}/${DATASET}/retrieved.jsonl"

    python evaluate.py \
      --dataset_name "$DATASET" \
      --gt_file "$GT_PATH" \
      --rerank_pred_file "$PRED_FILE" \
      --unrerank_pred_file "$BASELINE_FILE" \
      --k 10
    echo
  done
done
