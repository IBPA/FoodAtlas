#!/bin/bash

#SBATCH --job-name=grid_search
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=24:00:00

cd ..

# CUDA_VISIBLE_DEVICE=1, python -m food_atlas.entailment.experiments.run_grid_search \
#     outputs/data_generation/train_1.tsv \
#     outputs/data_generation/val.tsv \
#     biobert \
#     outputs/experiments/augmentation/aug_10 \
#     --random-seed 42

# python -m food_atlas.entailment.experiments.run_grid_search \
#     outputs/data_generation/train_1_orig.tsv \
#     outputs/data_generation/val.tsv \
#     biobert \
#     outputs/experiments/augmentation/orig \
#     --random-seed 42

CUDA_VISIBLE_DEVICES=3, python -m food_atlas.entailment.experiments.run_grid_search \
    outputs/data_generation/train_1_aug_5.tsv \
    outputs/data_generation/val.tsv \
    biobert \
    outputs/experiments/augmentation/aug_5 \
    --random-seed 42

cd scripts
