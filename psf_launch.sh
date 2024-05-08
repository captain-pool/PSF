#!/bin/bash -l

#$ -l gpus=10
#$ -pe omp 30
#$ -l gpu_memory=20G
#$ -e ./stderr.txt
#$ -o ./stdout.txt

module load miniconda
module load pytorch/1.9.0
module load cuda/12
conda activate psf
cd /project/otmesh/psf
python3 train_flow.py --dataroot ../shapenet/ --category chair
