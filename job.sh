#!/bin/bash
# This script is used to run the training and sampling processes for a machine learning model.
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --time=1-0 
#SBATCH -p general
#SBATCH -q public
#SBATCH -o=job.out
#SBATCH -e=job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anaray84@asu.edu
module load mamba/latest
mamba create -n myenv python=3.10
source activate myenv
pip install torch torchvision numpy scikit-learn tqdm matplotlib
python test.py --mode compute_centroids
python test.py --mode train --epochs 50
python sample.py --checkpoint_path checkpoints/image_gpt_epoch10.pth --centroids_path centroids_32.npy --index 0