#!/bin/bash

#SBATCH --job-name=FA_grid_search
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

ROUND=3

python -m food_atlas.entailment.run_train_data_merge \
    outputs/data_generation \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND \
    $ROUND \
    random_sample_each_bin \

cd scripts
