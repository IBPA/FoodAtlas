#!/bin/bash

#SBATCH --job-name=FA_predict
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

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_unlabeled_data_prediction \
    outputs/data_generation/2/random_sample_each_bin/to_predict_2.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/1/eval_best_model \

cd scripts
