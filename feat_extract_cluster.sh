#!/bin/bash -l
#
#$ -l gpus=1
#$ -pe omp 30
#$ -e ./stderr.txt
#$ -o ./stdout.txt


module load miniconda
module load pytorch
module load cuda/12

conda activate psf

cd /project/otmesh/psf

python3 -m datasets.feature_extractor --dataroot ../shapenet/ --category chair --nviews 25
