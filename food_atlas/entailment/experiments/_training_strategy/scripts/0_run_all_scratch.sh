#!/bin/bash

#SBATCH --job-name=FA_scratch
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

cd ~/git/FoodAtlas

CUDA=3
ROUND=3
AL=random_sample_each_bin
PATH_OUTPUT=/data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/scratch

python -m food_atlas.entailment.run_train_data_merge \
    outputs/data_generation \
    $PATH_OUTPUT \
    $ROUND \
    $AL \

CUDA_VISIBLE_DEVICES=${CUDA}, python -m food_atlas.entailment.run_grid_search \
    ${PATH_OUTPUT}/train_${ROUND}_merged.tsv \
    outputs/data_generation/val.tsv \
    biobert \
    ${PATH_OUTPUT}/grid_search \

CUDA_VISIBLE_DEVICES=${CUDA}, python -m food_atlas.entailment.run_best_model_evaluation \
    ${PATH_OUTPUT}/train_${ROUND}_merged.tsv \
    outputs/data_generation/val.tsv \
    outputs/data_generation/test.tsv \
    biobert \
    ${PATH_OUTPUT}/grid_search/grid_search_result_summary.csv \
    ${PATH_OUTPUT}/eval_best_model \

CUDA_VISIBLE_DEVICES=${CUDA}, python -m food_atlas.entailment.run_unlabeled_data_prediction \
    outputs/data_generation/$ROUND/$AL/to_predict_$ROUND.tsv \
    biobert \
    ${PATH_OUTPUT}/eval_best_model \
    --path-output-data-to-predict ${PATH_OUTPUT}/predicted_$ROUND.tsv \

cd ~/git/FoodAtlas/food_atlas/entailment/experiments/_training_strategy/scripts
