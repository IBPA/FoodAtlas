#!/bin/bash
set -e

dataset="annotations_predictions-above80_extdb_mesh_ncbi"
model="RotatE"

python run_test.py \
    --train_dir=../../../outputs/kgc/data/predict_extdb \
    --val_dir=../../../outputs/kgc/data/predict_extdb \
    --test_dir=../../../outputs/kgc/data/predict_extdb \
    --input_dir=../../../outputs/kgc/pykeen/$dataset/hpo/$model \
    --num_replications=5 \
    --output_dir=../../../outputs/kgc/pykeen/$dataset/hpo/$model \
    --is_production
