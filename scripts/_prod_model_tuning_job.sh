#!/bin/bash

#SBATCH --job-name=FA_prod
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/IBPA/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/IBPA/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=10-00:00:00

cd ..

PATH_DATA_FOLDS_DIR=outputs/data_processing/folds_for_prod_model
PATH_OUTPUT_DIR=/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/grid_search

echo "Running 10-fold CV for the production entailment model."
echo "PATH_DATA_FOLDS_DIR: $PATH_DATA_FOLDS_DIR"
echo "PATH_OUTPUT_DIR: $PATH_OUTPUT_DIR"
echo "FOLD_START: $1"
echo "FOLD_END: $2"

for (( fold=$1; fold<=$2; fold++ ))
do
    python -m food_atlas.entailment.run_grid_search \
        $PATH_DATA_FOLDS_DIR/fold_${fold}/train.tsv \
        $PATH_DATA_FOLDS_DIR/fold_${fold}/val.tsv \
        biobert \
        $PATH_OUTPUT_DIR/fold_${fold}/grid_search \
        --batch-sizes 16,32 \
        --learning-rates 2e-5,5e-5 \
        --nums-epochs 3,4 \
        --seeds 42
done
