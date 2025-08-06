#!/bin/bash

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen3-32B"
  "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  "deepseek/DeepSeek-V3-0324"
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
  "deepseek/DeepSeek-R1-0528"
)

GPU=$1     # e.g., H100 or B200
PREC=$2    # e.g., bf16, fp16

if [ -z "$GPU" ] || [ -z "$PREC" ]; then
  echo "Usage: ./batch_benchmark.sh H100 bf16"
  exit 1
fi

for MODEL in "${MODELS[@]}"; do
  echo "Running $MODEL on $GPU at $PREC precision"
  python roi_benchmark.py \
    --gpu "$GPU" \
    --model "$MODEL" \
    --precision "$PREC" \
    --skip-inference
done
