#!/bin/bash
set -e

# for multiple datasets, you can run the following
for dataset in annotations annotations_extdb annotations_mesh_ncbi annotations_predictions annotations_predictions_extdb annotations_predictions_extdb_mesh_ncbi
do
    for model in TransE TransD RotatE DistMult ER-MLP ComplEx
    do
        echo 'Dataset:' $dataset
        echo 'Model:' $model
        python run_test.py \
            --train_dir=../../../outputs/kgc/data/$dataset \
            --val_dir=../../../outputs/kgc/data \
            --test_dir=../../../outputs/kgc/data \
            --input_dir=../../../outputs/kgc/pykeen/$dataset/hpo/$model \
            --num_replications=5 \
            --output_dir=../../../outputs/kgc/pykeen/$dataset/hpo/$model \
            --generate_test_stats
        echo ''
        echo ''
        echo ''
    done
done
