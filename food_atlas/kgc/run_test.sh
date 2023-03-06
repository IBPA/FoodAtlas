#!/bin/bash
set -e

# for model in TransE TransD RotatE DistMult ER-MLP ComplEx TuckER
# do
#     echo 'Model:' $model

#     python run_test_using_best_model.py \
#         --input_kg_dir=../../outputs/kgc/data/annotations \
#         --val_test_dir=../../outputs/kgc/data \
#         --input_dir=../../outputs/kgc/pykeen/annotations/hpo/$model \
#         --num_replications=5 \
#         --output_dir=../../outputs/kgc/pykeen/annotations/hpo/$model
# done

# for model in TransE TransD RotatE DistMult ER-MLP ComplEx TuckER
# do
#     echo 'Model:' $model

#     python run_test_using_best_model.py \
#         --input_kg_dir=../../outputs/kgc/data/annotations_extdb \
#         --val_test_dir=../../outputs/kgc/data \
#         --input_dir=../../outputs/kgc/pykeen/annotations_extdb/hpo/$model \
#         --num_replications=5 \
#         --output_dir=../../outputs/kgc/pykeen/annotations_extdb/hpo/$model
# done

# for model in TransE TransD RotatE DistMult ER-MLP ComplEx TuckER
# do
#     echo 'Model:' $model

#     python run_test_using_best_model.py \
#         --input_kg_dir=../../outputs/kgc/data/annotations_mesh_ncbi \
#         --val_test_dir=../../outputs/kgc/data \
#         --input_dir=../../outputs/kgc/pykeen/annotations_mesh_ncbi/hpo/$model \
#         --num_replications=5 \
#         --output_dir=../../outputs/kgc/pykeen/annotations_mesh_ncbi/hpo/$model
# done

for model in TransD
do
    echo 'Model:' $model

    python run_test_using_best_model.py \
        --input_kg_dir=../../outputs/kgc/data/annotations_predictions_extdb_mesh_ncbi \
        --val_test_dir=../../outputs/kgc/data \
        --input_dir=../../outputs/kgc/pykeen/annotations_predictions_extdb_mesh_ncbi/hpo/$model \
        --num_replications=5 \
        --output_dir=../../outputs/kgc/pykeen/annotations_predictions_extdb_mesh_ncbi/hpo/$model
done
