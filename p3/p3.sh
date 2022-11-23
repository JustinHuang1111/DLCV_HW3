#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1 
#PBS -q ee
cd $PBS_O_WORKDIR
module load cuda/cuda-11.3/x86_64
source activate b09901062_env
# CUDA_VISIBLE_DEVICES = 3
python ./main.py ###你要跑的code
# python ./reference/inference.py ###你要跑的code
conda deactivate 
