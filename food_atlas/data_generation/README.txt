1. Generate PH pairs.

python query_and_generate_ph_pairs.py

Output files
- ../../outputs/data_generation/query_results.txt
- ../../outputs/data_generation/ph_pairs_{timestamp}.txt



2. Generate pre-annotation data for round 1.

python generate_pre_annotation.py \
    --round=1 \
    --pre_annotation_filepath=../../outputs/data_generation/1/pre_annotation_1.tsv \
    --to_predict_filepath=../../outputs/data_generation/1/to_predict_1.tsv

Output files
- ../../outputs/data_generation/1/pre_annotation_1.tsv
- ../../outputs/data_generation/val_pre_annotation.tsv
- ../../outputs/data_generation/test_pre_annotation.tsv
- ../../outputs/data_generation/1/to_predict_1.tsv



3. Annotate pre_annotation files. When finished, save the file names as below.

../../outputs/data_generation/1/pre_annotation_1.tsv -> ../../outputs/data_generation/1/post_annotation_1.tsv
../../outputs/data_generation/val_pre_annotation.tsv -> ../../outputs/data_generation/val_post_annotation.tsv
../../outputs/data_generation/test_pre_annotation.tsv -> ../../outputs/data_generation/test_post_annotation.tsv



4. Post process the annotation.

python post_process_annotation.py \
    --round=1 \
    --post_annotation_filepath=../../outputs/data_generation/1/post_annotation_1.tsv \
    --train_filepath=../../outputs/data_generation/1/train_1.tsv \
    --kg_output_dir=../../outputs/kg/1

Output files
- ../../outputs/data_generation/1/train_1.tsv
- ../../outputs/data_generation/val.tsv
- ../../outputs/data_generation/test.tsv
- ../../outputs/kg/1/kg.txt
- ../../outputs/kg/1/evidence.txt
- ../../outputs/kg/1/entities.txt
- ../../outputs/kg/1/relations.txt



5. Train the entailment model.

Save the output file as
- ../../outputs/data_generation/1/predicted_1.tsv



6. Generate round 2 pre annotation.

python generate_pre_annotation.py \
    --round=2 \
    --pre_annotation_filepath=../../outputs/data_generation/2/random_sample_each_bin/pre_annotation_2.tsv \
    --to_predict_filepath=../../outputs/data_generation/2/random_sample_each_bin/to_predict_2.tsv \
    --predicted_filepath=../../outputs/data_generation/1/predicted_1.tsv \
    --sampling_strategy=random_sample_each_bin

Output files
- ../../outputs/data_generation/2/random_sample_each_bin/pre_annotation_2.tsv
- ../../outputs/data_generation/2/random_sample_each_bin/to_predict_2.tsv


7. Annotate pre_annotation file. When finished, save the file names as below.

../../outputs/data_generation/2/random_sample_each_bin/pre_annotation_2.tsv -> ../../outputs/data_generation/2/random_sample_each_bin/post_annotation_2.tsv



8. Post process the annotation.

python post_process_annotation.py \
    --round=2 \
    --post_annotation_filepath=../../outputs/data_generation/2/random_sample_each_bin/post_annotation_2.tsv \
    --train_filepath=../../outputs/data_generation/2/random_sample_each_bin/train_2.tsv \
    --kg_output_dir=../../outputs/kg/2/random_sample_each_bin \
    --prev_kg_output_dir=../../outputs/kg/1

Output files
- ../../outputs/data_generation/2/random_sample_each_bin/train_2.tsv
- ../../outputs/kg/2/random_sample_each_bin/kg.txt
- ../../outputs/kg/2/random_sample_each_bin/evidence.txt
- ../../outputs/kg/2/random_sample_each_bin/entities.txt
- ../../outputs/kg/2/random_sample_each_bin/relations.txt



9. Train the entailment model.

Save the output file as
- ../../outputs/data_generation/2/random_sample_each_bin/predicted_2.tsv



10. Generate round 3 pre annotation.

python generate_pre_annotation.py \
    --round=3 \
    --pre_annotation_filepath=../../outputs/data_generation/3/random_sample_each_bin/pre_annotation_3.tsv \
    --to_predict_filepath=../../outputs/data_generation/3/random_sample_each_bin/to_predict_3.tsv \
    --predicted_filepath=../../outputs/data_generation/2/random_sample_each_bin/predicted_2.tsv \
    --sampling_strategy=random_sample_each_bin


























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

