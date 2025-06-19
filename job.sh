#!/bin/bash
# This script is used to run the training and sampling processes for a machine learning model.
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --time=0-8 
#SBATCH -p general
#SBATCH -q public
#SBATCH -o=job.out
#SBATCH -e=job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anaray84@asu.edu
module load mamba/latest
source activate myenv
python src/train.py
python src/sample.py --ckpt_num 1
python src/sample.py --ckpt_num 2
python src/sample.py --ckpt_num 3
python src/sample.py --ckpt_num 4
python src/sample.py --ckpt_num 5
python src/sample.py --ckpt_num 6
python src/sample.py --ckpt_num 7
python src/sample.py --ckpt_num 8
python src/sample.py --ckpt_num 9
python src/sample.py --ckpt_num 10
python src/sample.py --ckpt_num 11
python src/sample.py --ckpt_num 12