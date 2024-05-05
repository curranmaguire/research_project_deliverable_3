#!/bin/bash

#SBATCH --job-name=DCG_train
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#sbatch --mail-type=ALL
#SBATCH --mail-user=cgmj52@durham.ac.uk

conda activate lewis


python DCG_classifier_train.py
python DCG_bart_train.py
python inference.py