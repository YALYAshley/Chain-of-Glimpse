#!/bin/bash
#PBS -N Qwen7B_RL
#PBS -l select=1:ncpus=16:ngpus=5:mem=300gb
#PBS -j oe
#PBS -o ./202601RL-7b.txt
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
export CUDA_VISIBLE_DEVICES=4
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
bash .//scripts/vllm/serve_qwen.sh &
VLLM_PID=$!
sleep 180
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd .//src/trainer/rl
bash .//src/trainer/rl/examples/run_vigorl_7b.sh

kill $VLLM_PID || true
echo "Job finished at $(date)"
