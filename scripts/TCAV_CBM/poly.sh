#!/bin/bash

#SBATCH -G a100:1
#SBATCH -c 32
#SBATCH --mem 24G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 2-00:00:00   # time in d-hh:mm:ss

module purge
module load mamba/latest
source activate BBTCAV

cd ../../

python -m tests.CelebA_TCAV --config local_configs/TCAV_CBM/model_5.yaml
python -m tests.CelebA_TCAV --config local_configs/TCAV_CBM/model_6.yaml
python -m tests.CelebA_TCAV --config local_configs/TCAV_CBM/model_7.yaml