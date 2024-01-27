#!/bin/bash
#SBATCH --partition=gpu3-2
#SBATCH --nodelist=g3013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64

cd /home/zhaoyue/Grounding/CLIP

lr=5e-5
max_epoch=20
batch_size=8
save_epoch=5