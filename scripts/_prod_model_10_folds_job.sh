#!/bin/bash

#SBATCH --job-name=FA_fold
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
PATH_OUTPUT_DIR=/data/lfz/projects/FoodAtlas/outputs/entailment_model/10_folds_analysis

echo "Running 10-fold CV for the analysis."
echo "PATH_DATA_FOLDS_DIR: $PATH_DATA_FOLDS_DIR"
echo "PATH_OUTPUT_DIR: $PATH_OUTPUT_DIR"
echo "SEED_START: $1"
echo "SEED_END: $2"

for (( seed=$1; seed<=$2; seed++ ))
do
    for fold in {0..9}
    do
        python -m food_atlas.entailment.train_prod_model \
            $PATH_DATA_FOLDS_DIR/fold_${fold}/train.tsv \
            $PATH_DATA_FOLDS_DIR/fold_${fold}/val.tsv \
            biobert \
            $PATH_OUTPUT_DIR/fold_${fold} \
            --seeds $seed
    done
done
