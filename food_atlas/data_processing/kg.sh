#!/bin/bash
set -e

echo "Annotation training pool to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/train_pool.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated \
    --nb_workers=10

echo
echo "Annotation validation to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/val.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated \
    --nb_workers=10

echo
echo "Annotation test to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/test.tsv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations \
    --mode=annotated \
    --nb_workers=10

echo
echo "Predictions to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/predicted.csv \
    --input_kg_dir=../../outputs/kg/annotations/ \
    --output_kg_dir=../../outputs/kg/annotations_predictions \
    --mode=predicted \
    --nb_workers=10

echo
echo "LitSense additional predictions to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/ph_pairs_20230119_214855_predicted.tsv \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --output_kg_dir=../../outputs/kg/annotations_predictions \
    --mode=predicted \
    --nb_workers=2

echo
echo "Phenol-Explorer Predictions to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/ph_pairs_20230111_224704_predicted.tsv \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --output_kg_dir=../../outputs/kg/annotations_predictions \
    --mode=predicted \
    --nb_workers=5

echo
echo "Frida Predictions to KG..."
python generate_kg_from_ph_pairs.py \
    --input_filepath=../../outputs/data_processing/ph_pairs_20230112_114749_predicted.tsv \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --output_kg_dir=../../outputs/kg/annotations_predictions \
    --mode=predicted \
    --nb_workers=5

echo
echo "Frida to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --output_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --external_db_filepath=../../data/Frida/frida.tsv \
    --external_db_name=Frida

echo
echo "Phenol-Explorer to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --output_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --external_db_filepath=../../data/Phenol-Explorer/phenol_explorer.tsv \
    --external_db_name=Phenol-Explorer

echo
echo "FDC to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --output_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --external_db_filepath=../../data/FDC/fdc.tsv \
    --external_db_name=FDC

echo
echo "MeSH to KG..."
python merge_mesh.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --output_kg_dir=../../outputs/kg/annotations_predictions_extdb_mesh \
    --use_pkl \
    --nb_workers=10

echo
echo "NCBI taxonomy to KG..."
python merge_ncbi_taxonomy.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb_mesh \
    --output_kg_dir=../../outputs/kg/annotations_predictions_extdb_mesh_ncbi \
    --nb_workers=10

echo
echo "Final KG parsing..."
python parse_final_kg.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb_mesh_ncbi \
    --output_dir=../../outputs/backend_data/v0.1
