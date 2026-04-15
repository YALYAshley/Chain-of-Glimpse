#!/bin/bash
#PBS -N Qwen7B_MCTS
#PBS -l select=1:ncpus=16:ngpus=2:mem=300gb
#PBS -j oe
#PBS -o 202601sft_7b.txt
#PBS -P personal
#PBS -q normal
#PBS -l walltime=20:00:00

set -e
echo "Job started on $(hostname) at $(date)"

cd /home/users/ntu/n2505667/grounded-rl-main
module load miniforge3/25.3.1
conda activate chain-of-glimpse

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

################################
# vLLM (GPUs 0–3)
################################
export CUDA_VISIBLE_DEVICES=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
bash /vllm/serve_qwen_7b.sh &
VLLM_PID=$!
sleep 180
nvidia-smi

export CUDA_VISIBLE_DEVICES=0
cd /src/trainer/offline
bash /src/trainer/offline/examples/train_qwen2_5_vl_7b_sft.sh

kill $VLLM_PID || true
echo "Job finished at $(date)"
