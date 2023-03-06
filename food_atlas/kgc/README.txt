# Generate val/test data using the annotations KG.
# We also generate the training data.
python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations \
    --full_kg_dir=../../outputs/kg/annotations_predictions_extdb_extdbpred_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations \
    --is_initial

# We now generate training data for annotations_predictions KG.
# We don't have to generate val/test set, so do not set --is_initial.
python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations_predictions

python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations_extdb \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations_extdb

python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations_mesh_ncbi

python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations_predictions_extdb

python generate_data.py \
    --input_kg_dir=../../outputs/kg/annotations_predictions_extdb_mesh_ncbi \
    --val_test_dir=../../outputs/kgc/data \
    --output_dir=../../outputs/kgc/data/annotations_predictions_extdb_mesh_ncbi



# Run hpo for different models.
python run_hpo.py \
    --dataset_dir=../../outputs/kgc/data/annotations \
    --output_dir=../../outputs/kgc/pykeen/annotations/hpo \
    --models=TransE,TransD,RotatE,DistMult,ER-MLP,ComplEx,TuckER

# Run the best model on the test set and generate statistics. Repeat it for all models.
python run_test_using_best_model.py \
    --input_kg_dir=../../outputs/kgc/data/annotations \
    --val_test_dir=../../outputs/kgc/data \
    --input_dir=../../outputs/kgc/pykeen/annotations/hpo/TransD \
    --num_replications=5 \
    --output_dir=../../outputs/kgc/pykeen/annotations/hpo/TransD
