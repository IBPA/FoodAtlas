#!/bin/bash

set -e

#SBATCH --job-name=FA_grid_search
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/IBPA/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/IBPA/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=20-00:00:00

# TODO: Change this to your own path with enough disk space.
#   This is where all the training related results will be saved. Be prepared
#   for the disk space usage to be around 2 TB.
PATH_OUTPUT_ROOT=/data/lfz/projects/FoodAtlas/outputs

cd ..

echo "AL strategy: $1"
echo "Run: $2"
echo "Round: $3"

AL=$1
RUN=$2
ROUND=$3

# Compute random seed based on the inputs to make sure no random seed is repeated.
if [ $1 == 'uncertain' ]
then
    RANDOM_SEED=1
elif [ $1 == 'certain_pos' ]
then
    RANDOM_SEED=2
elif [ $1 == 'stratified' ]
then
    RANDOM_SEED=3
elif [ $1 == 'random' ]
then
    RANDOM_SEED=4
fi

RANDOM_SEED=$((RANDOM_SEED * 10000 + RUN))

echo $RANDOM_SEED

PATH_OUTPUT=$PATH_OUTPUT_ROOT/entailment_model/$AL/run_${RUN}/round_${ROUND}
PATH_TRAIN_POOL=outputs/data_generation/train_pool.tsv
PATH_VAL=outputs/data_generation/val.tsv
PATH_TEST=outputs/data_generation/test.tsv

cd food_atlas/data_generation
python prepare_training_data.py \
    --sampling_strategy=$AL \
    --run=$RUN \
    --round=$ROUND \
    --total_rounds=10 \
    --random_state=$RANDOM_SEED \
    --train_pool_filepath=../../$PATH_TRAIN_POOL
cd ../..

python -m food_atlas.entailment.run_grid_search \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/train.tsv \
    $PATH_VAL \
    biobert \
    $PATH_OUTPUT/grid_search \
    --batch-sizes 16,32 \
    --learning-rates 2e-5,5e-5 \
    --nums-epochs 3,4 \
    --seeds $RANDOM_SEED \

python -m food_atlas.entailment.run_best_model_evaluation \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/train.tsv \
    $PATH_VAL \
    $PATH_TEST \
    biobert \
    $PATH_OUTPUT/grid_search/grid_search_result_summary.csv \
    $PATH_OUTPUT/eval_best_model \
    --seeds $RANDOM_SEED \

python -m food_atlas.entailment.run_unlabeled_data_prediction \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/to_predict.tsv \
    biobert \
    $PATH_OUTPUT/eval_best_model \
    --path-output-data-to-predict outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/predicted.tsv \

python -m food_atlas.entailment.run_unlabeled_data_prediction \
    $PATH_TEST \
    biobert \
    $PATH_OUTPUT/eval_best_model \
    --path-output-data-to-predict outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/test_probs.tsv \

cd scripts
