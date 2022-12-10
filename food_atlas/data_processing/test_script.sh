#!/bin/bash
set -e

# --sampling_strategy can be (uncertain|certain_pos|stratified)

for run in $(seq 1 2);
do
    for round in $(seq 1 10);
    do
        python prepare_training_data.py \
            --sampling_strategy=stratified \
            --run=$run \
            --round=$round \
            --total_rounds=10 \
            --random_state=530

        python simulate_prediction.py \
            --sampling_strategy=stratified \
            --run=$run \
            --round=$round \
            --random_state=530
    done
done
