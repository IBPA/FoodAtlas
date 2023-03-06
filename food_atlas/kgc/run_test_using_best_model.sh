#!/bin/bash
set -e

# # for specific dataset (annotations in this case), you can run the following
# for model in TransE TransD RotatE DistMult ER-MLP ComplEx TuckER
# do
#     echo 'Model:' $model
#     python run_test_using_best_model.py \
#         --input_kg_dir=../../outputs/kgc/data/annotations \
#         --val_test_dir=../../outputs/kgc/data \
#         --input_dir=../../outputs/kgc/pykeen/annotations/hpo/$model \
#         --num_replications=5 \
#         --output_dir=../../outputs/kgc/pykeen/annotations/hpo/$model
#     echo ''
#     echo ''
#     echo ''
# done

# for multiple datasets, you can run the following
for dataset in annotations_predictions annotations_extdb annotations_mesh_ncbi annotations_predictions_extdb annotations_predictions_extdb_mesh_ncbi
do
    for model in TransE TransD RotatE DistMult ER-MLP ComplEx TuckER
    do
        echo 'Dataset:' $dataset
        echo 'Model:' $model
        python run_test_using_best_model.py \
            --input_kg_dir=../../outputs/kgc/data/$dataset \
            --val_test_dir=../../outputs/kgc/data \
            --input_dir=../../outputs/kgc/pykeen/$dataset/hpo/$model \
            --num_replications=5 \
            --output_dir=../../outputs/kgc/pykeen/$dataset/hpo/$model
        echo ''
        echo ''
        echo ''
    done
done
