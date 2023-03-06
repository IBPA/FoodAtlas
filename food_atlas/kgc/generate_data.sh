#!/bin/bash
set -e

# Generate initial KG as well as head-out val/test set.
python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations \
    --full_kg_dir=../../outputs/kg/annotations_predictions_extdb_extdbpred_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations \
    --is_initial

# Now that the held-out val/test is generated, we generate other data for KGC.
for dataset in annotations_predictions annotations_extdb annotations_mesh_ncbi annotations_predictions_extdb annotations_predictions_extdb_mesh_ncbi
do
    echo 'Dataset:' $dataset
    python generate_data.py \
        --input_kg_dir=../../outputs/kg/$dataset \
        --val_test_dir=../../outputs/kgc/data \
        --output_dir=../../outputs/kgc/data/$dataset
    echo ''
    echo ''
    echo ''
done
