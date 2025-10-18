#!/bin/bash
#SBATCH --job-name=ap_beauty          # 作业名
#SBATCH --partition=rtx8000               
#SBATCH --gres=gpu:rtx8000:1             # 1×A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=pr_119_tandon_priority
#SBATCH --output=logs/%x_%j.out       # 标准输出
#SBATCH --error=logs/%x_%j.err        # 标准错误
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=yh4663@nyu.edu
#SBATCH --requeue
# ───── 1. 载模块 & 环境 ─────
module purge
module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate actionpiece

# ───── 2. 数据缓存目录 ─────
CACHE_DIR=$SCRATCH/datasets           # 自己可写的位置
mkdir -p "$CACHE_DIR"

# ───── 3. 训练命令 ─────
CUDA_VISIBLE_DEVICES=0 python main.py \
  --category=Beauty \
  --weight_decay=0.15 \
  --lr=0.001 \
  --n_hash_buckets=64 \
  --cache_dir="$CACHE_DIR"

conda deactivate