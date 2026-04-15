#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# check if argument is provided, otherwise use default
CHECKPOINT_PATH="./LM-Model/Qwen2.5-VL-7B-Instruct"
NUM_GPUS=1
PORT=19012

# Check if checkpoint path exists
if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
    exit 1
fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"

mkdir -p vllm_logs

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization 0.75 \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --limit-mm-per-prompt "image=15" \
    --mm-processor-kwargs '{"max_pixels":65536,"min_pixels":1024}' \
    --max-model-len 32768 \
    --api-key "qwen" > "./vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi