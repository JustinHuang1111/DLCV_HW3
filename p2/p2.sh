#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1 
#PBS -q ee
cd $PBS_O_WORKDIR
module load cuda/cuda-11.3/x86_64
source activate b09901062_env
# CUDA_VISIBLE_DEVICES = 3
bash ../hw3_1.sh ../hw3_data/p1_data/val ../hw3_data/p1_data/id2label.json ../pred.csv
# python ./reference/evaluate.py --pred_file /home/eegroup/ee50526/b09901062/hw3-JustinHuang1111/p2/reference/json_out/newEval.json ###你要跑的code
# python ./inference.py --imgpath ../hw3_data/p2_data/images/val --outpath ./reference/json_out/newEval.json --model ./reference/ckpt/1711.0215/1117-0215_epsilon_50_best.pth  ###你要跑的code
conda deactivate 
