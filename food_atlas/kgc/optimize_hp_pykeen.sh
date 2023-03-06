#!/bin/bash
set -e

# python run_hpo.py \
#     --train_dir=../../outputs/kgc/data/annotations \
#     --val_test_dir=../../outputs/kgc/data \
#     --output_dir=../../outputs/kgc/pykeen/annotations/hpo \
#     --models=TransE,TransD,RotatE,DistMult,ER-MLP,ComplEx,TuckER

python run_hpo.py \
    --train_dir=../../outputs/kgc/data/annotations_extdb \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/pykeen/annotations_extdb/hpo \
    --models=TransE,TransD,RotatE,DistMult,ER-MLP,ComplEx,TuckER

python run_hpo.py \
    --train_dir=../../outputs/kgc/data/annotations_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/pykeen/annotations_mesh_ncbi/hpo \
    --models=TransE,TransD,RotatE,DistMult,ER-MLP,ComplEx,TuckER
