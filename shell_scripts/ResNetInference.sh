#!/bin/bash

#PBS -N benchmark_ResNet
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

echo "--- Inference Job of ResNet Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 3. Run the evaluation script
python benchmarkResNet.py

echo "--- Job Finished at $(date) ---"
