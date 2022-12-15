#!/bin/bash
set -e

# echo "Annotation training pool to KG..."
# python generate_kg.py \
#     --input_filepath=../../outputs/data_processing/train_pool.tsv \
#     --input_kg_dir=../../outputs/kg/annotations/ \
#     --output_kg_dir=../../outputs/kg/annotations \
#     --mode=annotated

# echo
# echo "Annotation validation to KG..."
# python generate_kg.py \
#     --input_filepath=../../outputs/data_processing/val.tsv \
#     --input_kg_dir=../../outputs/kg/annotations/ \
#     --output_kg_dir=../../outputs/kg/annotations \
#     --mode=annotated

# echo
# echo "Annotation test to KG..."
# python generate_kg.py \
#     --input_filepath=../../outputs/data_processing/test.tsv \
#     --input_kg_dir=../../outputs/kg/annotations/ \
#     --output_kg_dir=../../outputs/kg/annotations \
#     --mode=annotated

# echo
# echo "Predictions to KG..."
# python generate_kg.py \
#     --input_filepath=../../outputs/data_processing/predicted.csv \
#     --input_kg_dir=../../outputs/kg/annotations/ \
#     --output_kg_dir=../../outputs/kg/annotations_predictions \
#     --mode=predicted

# # Merge MeSH first so that we can get the CAS IDs for each chemical
# # that will be used later for merging PubChem as well as external DBs
# echo
# echo "MeSH to KG..."
# python merge_mesh.py \
#     --input_kg_dir=../../outputs/kg/annotations_predictions \
#     --output_kg_dir=../../outputs/kg/annotations_predictions_mesh \
#     --use_pkl

# echo
# echo "NCBI taxonomy to KG..."
# python merge_ncbi_taxonomy.py \
#     --input_kg_dir=../../outputs/kg/annotations_predictions_mesh \
#     --output_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi

# # Adds additional chemical IDs using PubChem
# echo
# echo "PubChem to KG..."
# python merge_pubchem.py \
#     --input_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi/ \
#     --output_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi_pubchem \
#     --use_pkl

echo
echo "Frida to KG..."
python merge_external_dbs.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi_pubchem/ \
    --output_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi_pubchem_frida \
    --external_db_filepath=../../outputs/merge_dbs/food_chemical_triples/frida.tsv \
    --external_db_name=Frida

# echo
# echo "Phenol-Explorer to KG..."
# python merge_external_dbs.py \
#     --input_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi_pubchem_frida/ \
#     --output_kg_dir=../../outputs/kg/annotations_predictions_mesh_ncbi_pubchem_frida_phenolexplorer \
#     --external_db_filepath=../../outputs/merge_dbs/food_chemical_triples/phenol_explorer.tsv \
#     --external_db_name=Phenol-Explorer



# do sanity check
# update chemical and taxonomy names to official names
