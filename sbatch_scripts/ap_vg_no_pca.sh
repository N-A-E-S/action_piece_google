#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/Collaborative-Learning-with-Action-aware-Image-text-Representation-Optimization/

CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --rand_seed=67 \
    --lr=0.001 \
    --weight_decay=0.07 \
    --n_hash_buckets=256 \
    --d_model=256 \
    --d_ff=2048 \
    --dataset=AmazonReviews2018

conda deactivate