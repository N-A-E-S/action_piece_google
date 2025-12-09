#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=80:00:00
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/Collaborative-Learning-with-Action-aware-Image-text-Representation-Optimization
python build_vocab.py   \
    --category=Musical_Instruments          \
    --multimodal.enable=false             \
    --n_hash_buckets=256 \
    --dataset=AmazonReviews2018

conda deactivate