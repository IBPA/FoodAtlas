#!/bin/bash
set -e

echo "Annotation training pool to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/train_pool.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated

echo
echo "Annotation validation to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/val.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated

echo
echo "Annotation test to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/test.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated

echo
echo "Predictions to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/predicted.csv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations_predictions \
    --mode=predicted

echo
echo "Frida to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --output_kg_dir=../../outputs/kg/annotations_predictions_frida \
    --external_db_filepath=../../data/Frida/frida.tsv \
    --external_db_name=Frida

echo
echo "Phenol-Explorer to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_frida \
    --output_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer \
    --external_db_filepath=../../data/Phenol-Explorer/phenol_explorer.tsv \
    --external_db_name=Phenol-Explorer

echo
echo "MeSH to KG..."
python merge_mesh.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer \
    --output_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer_mesh \
    --use_pkl

echo
echo "NCBI taxonomy to KG..."
python merge_ncbi_taxonomy.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer_mesh \
    --output_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer_mesh_ncbi

echo
echo "Final KG parsing..."
python parse_final_kg.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_frida_phenolexplorer_mesh_ncbi \
    --output_dir=../../outputs/backend_data/v0.1
