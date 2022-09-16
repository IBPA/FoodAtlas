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

python -m food_atlas.entailment.experiments.run_grid_search \
    outputs/data_generation/train_1.tsv \
    outputs/data_generation/val.tsv \
    outputs/round_1/entailment_model \
    biobert \
    --random-seed 42

cd scripts
