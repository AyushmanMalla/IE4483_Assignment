#!/bin/bash

#PBS -N benchmark_ViT
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

echo "--- Inference Job of ViT Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 3. Run the evaluation script
python benchmarkViT.py

echo "--- Job Finished at $(date) ---"
