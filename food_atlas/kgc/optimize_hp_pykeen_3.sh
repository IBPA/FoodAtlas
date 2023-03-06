#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=1 python run_hpo.py \
    --train_dir=../../outputs/kgc/data/annotations_predictions_extdb_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/pykeen/annotations_predictions_extdb_mesh_ncbi/hpo \
    --models=RotatE,DistMult
