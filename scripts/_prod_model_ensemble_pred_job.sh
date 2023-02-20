#!/bin/bash

#SBATCH --job-name=FA_prod_pred
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

DATA_TO_PREDICT=ph_pairs_20230118_144448.txt
# PATH_DATA_TO_PREDICT=outputs/data_processing/ph_pairs_20230111_224704.txt
# PATH_DATA_TO_PREDICT=outputs/data_processing/ph_pairs_20230112_114749.txt
PATH_DATA_TO_PREDICT=outputs/data_processing/$DATA_TO_PREDICT
PATH_MODEL_STATE_ROOT=/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/ensemble
PATH_OUTPUT=/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/ensemble

echo "Running ensemble production entailment model predictions."
echo "PATH_DATA_TO_PREDICT: $PATH_DATA_TO_PREDICT"

for (( seed=$1; seed<=$2; seed++ ))
do
    echo "PATH_OUTPUT: $PATH_OUTPUT/$seed/seed_${seed}/predicted_${DATA_TO_PREDICT}.tsv"
    python -m food_atlas.entailment.run_unlabeled_data_prediction_for_ensemble \
        $PATH_DATA_TO_PREDICT \
        biobert \
        $PATH_MODEL_STATE_ROOT/$seed/seed_${seed}/model_state.pt \
        --path-output-data-to-predict $PATH_OUTPUT/$seed/seed_${seed}/predicted_${DATA_TO_PREDICT}.tsv
done
