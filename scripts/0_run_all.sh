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
AL=random_sample_each_bin

python -m food_atlas.entailment.run_train_data_merge \
    outputs/data_generation \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND \
    $ROUND \
    $AL \

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_grid_search \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/train_${ROUND}_merged.tsv \
    outputs/data_generation/val.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/grid_search \

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_best_model_evaluation \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/train_${ROUND}_merged.tsv \
    outputs/data_generation/val.tsv \
    outputs/data_generation/test.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/grid_search/grid_search_result_summary.csv \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/eval_best_model \

CUDA_VISIBLE_DEVICES=0, python -m food_atlas.entailment.run_unlabeled_data_prediction \
    outputs/data_generation/$ROUND/$AL/to_predict_$ROUND.tsv \
    biobert \
    /data/lfz/projects/FoodAtlas/outputs/entailment_model/$ROUND/eval_best_model \
    --path-output-data-to-predict outputs/data_generation/$ROUND/$AL/predicted_$ROUND.tsv \

cd scripts
