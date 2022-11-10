1. Generate PH pairs.

python query_and_generate_ph_pairs.py

Output files
- ../../outputs/data_generation/query_results.txt
- ../../outputs/data_generation/ph_pairs_{timestamp}.txt



2. Generate pre-annotation data for training pool.

python generate_pre_annotation.py \
    --train_pre_annotation_filepath=../../outputs/data_generation/train_pool_pre_annotation.tsv

Output files
- ../../outputs/data_generation/train_pool_pre_annotation.tsv
- ../../outputs/data_generation/val_pre_annotation.tsv
- ../../outputs/data_generation/test_pre_annotation.tsv



3. Annotate pre_annotation files. When finished, save the file names as below.

../../outputs/data_generation/train_pool_pre_annotation.tsv-> ../../outputs/data_generation/train_pool_post_annotation.tsv
../../outputs/data_generation/val_pre_annotation.tsv -> ../../outputs/data_generation/val_post_annotation.tsv
../../outputs/data_generation/test_pre_annotation.tsv -> ../../outputs/data_generation/test_post_annotation.tsv



4. Post process the annotation.

python post_process_annotation.py \
    --train_post_annotation_filepath=../../outputs/data_generation/train_pool_post_annotation.tsv \
    --train_filepath=../../outputs/data_generation/train_pool.tsv

Output files
- ../../outputs/data_generation/train_pool.tsv
- ../../outputs/data_generation/val.tsv
- ../../outputs/data_generation/test.tsv



5. Train the entailment model.






X. Annotations to KG.

python generate_kg.py \
    --input_filepath=../../outputs/data_generation/train_pool.tsv \
    --output_dir=../../outputs/kg/annotations_only \
    --mode=annotated

python generate_kg.py \
    --input_filepath=../../outputs/data_generation/val.tsv \
    --output_dir=../../outputs/kg/annotations_only \
    --mode=annotated

python generate_kg.py \
    --input_filepath=../../outputs/data_generation/test.tsv \
    --output_dir=../../outputs/kg/annotations_only \
    --mode=annotated

Modified files
- ../../outputs/kg/annotations_only/kg.txt
- ../../outputs/kg/annotations_only/evidence.txt
- ../../outputs/kg/annotations_only/entities.txt
- ../../outputs/kg/annotations_only/relations.txt



X. MESH KG enrichment

python merge_mesh.py \
    --input_kg_dir=../../outputs/kg/annotations_only \
    --output_kg_dir=../../outputs/kg/annotations_mesh

Modified files
- ../../outputs/kg/annotations_only/kg.txt
- ../../outputs/kg/annotations_only/evidence.txt
- ../../outputs/kg/annotations_only/entities.txt
- ../../outputs/kg/annotations_only/relations.txt







6. (Optional) Add the entailment model predictions to the KG.
python add_model_predictions_to_kg.py --round=1

Modified files
- ../../outputs/kg/1/kg.txt
- ../../outputs/kg/1/evidence.txt
- ../../outputs/kg/1/entities.txt
- ../../outputs/kg/1/relations.txt



7. (Optional) Add NCBI taxonomy to the KG.
python merge_ncbi_taxonomy.py --round=1

Modified files
- ../../outputs/kg/1/kg.txt
- ../../outputs/kg/1/evidence.txt
- ../../outputs/kg/1/entities.txt
- ../../outputs/kg/1/relations.txt


