#!/bin/bash
#PBS -N Qwen3B_RL
#PBS -l select=1:ncpus=16:ngpus=3:mem=300gb
#PBS -j oe
#PBS -o /202601RL.txt
#PBS -P personal
#PBS -q normal
#PBS -l walltime=20:00:00

set -e
echo "Job started on $(hostname) at $(date)"

cd ./
module load miniforge3/25.3.1
conda activate chain-of-glimpse

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

################################
# vLLM (GPU)
################################
export CUDA_VISIBLE_DEVICES=2
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
bash .//scripts/vllm/serve_qwen.sh &
VLLM_PID=$!
sleep 180
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1
cd .//src/trainer/rl
bash .//src/trainer/rl/examples/run_vigorl_test.sh

kill $VLLM_PID || true
echo "Job finished at $(date)"
