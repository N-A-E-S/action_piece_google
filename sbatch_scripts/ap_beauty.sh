#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=28:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/Collaborative-Learning-with-Action-aware-Image-text-Representation-Optimization/


CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Beauty \
    --rand_seed=67 \
    --lr=0.001 \
    --weight_decay=0.15 \
    --n_hash_buckets=64 \
    --train_batch_size=256 \
    --epochs=200 \
    --patience=20 \
    --warmup_steps=10000 \
    --dropout_rate=0.1 \
    --d_model=128 \
    --d_ff=1024 \
    --num_layers=4 \
    --num_heads=6 \
    --d_kv=64 \
    --actionpiece_vocab_size=40000 \
    --num_beams=50 \
    --n_inference_ensemble=5

conda deactivate