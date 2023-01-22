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

PATH_DATA_TRAIN=outputs/data_processing/train_prod.tsv
PATH_DATA_TO_PREDICT=outputs/data_processing/to_predict.tsv
PATH_OUTPUT_DIR=/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/ensemble

echo "Running ensemble production entailment model."
echo "PATH_DATA_TRAIN: $PATH_DATA_TRAIN"
echo "PATH_DATA_TO_PREDICT: $PATH_DATA_TO_PREDICT"
echo "PATH_OUTPUT_DIR: $PATH_OUTPUT_DIR"
echo "SEED_START: $1"
echo "SEED_END: $2"

for (( seed=$1; seed<=$2; seed++ ))
do
    python -m food_atlas.entailment.train_prod_model \
        $PATH_DATA_TRAIN \
        $PATH_DATA_TO_PREDICT \
        biobert \
        $PATH_OUTPUT_DIR/$seed \
        --seeds $seed
done
