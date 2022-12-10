1. Generate PH pairs.

python query_and_generate_ph_pairs.py

Output files
- ../../outputs/data_processing/query_results.txt
- ../../outputs/data_processing/ph_pairs_{timestamp}.txt



2. Generate pre-annotation data for training pool.

python generate_pre_annotation.py \
    --train_pre_annotation_filepath=../../outputs/data_processing/train_pool_pre_annotation.tsv

Output files
- ../../outputs/data_processing/train_pool_pre_annotation.tsv
- ../../outputs/data_processing/val_pre_annotation.tsv
- ../../outputs/data_processing/test_pre_annotation.tsv
- ../../outputs/data_processing/to_predict.tsv



3. Annotate pre_annotation files. When finished, save the file names as below.

../../outputs/data_processing/train_pool_pre_annotation.tsv-> ../../outputs/data_processing/train_pool_post_annotation.tsv
../../outputs/data_processing/val_pre_annotation.tsv -> ../../outputs/data_processing/val_post_annotation.tsv
../../outputs/data_processing/test_pre_annotation.tsv -> ../../outputs/data_processing/test_post_annotation.tsv



4. Post process the annotation.

python post_process_annotation.py \
    --train_post_annotation_filepath=../../outputs/data_processing/train_pool_post_annotation.tsv \
    --train_filepath=../../outputs/data_processing/train_pool.tsv

Output files
- ../../outputs/data_processing/train_pool.tsv
- ../../outputs/data_processing/val.tsv
- ../../outputs/data_processing/test.tsv



5. Train the entailment model.






X. Annotations to KG.

python generate_kg.py \
    --input_filepath=../../outputs/data_processing/train_pool.tsv \
    --output_dir=../../outputs/kg/annotations \
    --mode=annotated

python generate_kg.py \
    --input_filepath=../../outputs/data_processing/val.tsv \
    --output_dir=../../outputs/kg/annotations \
    --mode=annotated

python generate_kg.py \
    --input_filepath=../../outputs/data_processing/test.tsv \
    --output_dir=../../outputs/kg/annotations \
    --mode=annotated



X. MESH KG enrichment

python merge_mesh.py \
    --input_kg_dir=../../outputs/kg/annotations \
    --output_kg_dir=../../outputs/kg/annotations_mesh \
    --use_pkl



X. Add NCBI taxonomy to the KG.
python merge_ncbi_taxonomy.py \
    --input_kg_dir=../../outputs/kg/annotations_mesh \
    --output_kg_dir=../../outputs/kg/annotations_mesh_ncbi



X. Generate k-folds for training the deploy entailment model.
python generate_folds.py \
    --input_train_filepath=../../outputs/data_processing/train_pool.tsv \
    --input_val_filepath=../../outputs/data_processing/val.tsv \
    --input_test_filepath=../../outputs/data_processing/test.tsv \
    --output_dir=../../outputs/data_processing/folds_for_prod_model











X. Add InChI and InChIKey to the KG using CAS.
python merge_inchi.py \
    --input_kg_dir=../../outputs/kg/annotations_mesh_ncbi/ \
    --output_kg_dir=../../outputs/kg/annotations_mesh_ncbi_inchi



X. (Optional) Add the entailment model predictions to the KG.
python add_model_predictions_to_kg.py --round=1
