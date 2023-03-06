#!/bin/bash
set -e

for dataset in annotations_predictions annotations_extdb annotations_mesh_ncbi annotations_predictions_extdb annotations_predictions_extdb_mesh_ncbi
do
    echo 'Dataset:' $dataset
    python run_hpo.py \
        --train_dir=../../outputs/kgc/data/$dataset \
        --val_test_dir=../../outputs/kgc/data \
        --output_dir=../../outputs/kgc/pykeen/$dataset/hpo \
        --models=TransE,TransD,RotatE,DistMult,ER-MLP,ComplEx,TuckER
    echo ''
    echo ''
    echo ''
done
