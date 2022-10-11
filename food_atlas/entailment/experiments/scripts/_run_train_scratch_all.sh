#!/bin/bash

#SBATCH --job-name=scratch_all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=48:00:00

cd ..

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_grid_search \
    tests/train_1_2.tsv \
    outputs/data_generation/val.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/2/scratch_all \

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_best_model_evaluation \
    tests/train_1_2.tsv \
    outputs/data_generation/val.tsv \
    outputs/data_generation/test.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/2/scratch_all \

cd scripts
