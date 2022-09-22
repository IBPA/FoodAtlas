1. Run query_and_generate_ph_pairs.py

Output files
- ../../outputs/data_generation/query_results.txt
- ../../outputs/data_generation/ph_pairs_{timestamp}.txt

2. Run generate_pre_annotation.py --round=1

Output files
- ../../outputs/data_generation/pre_annotation_1.tsv
- ../../outputs/data_generation/val_pre_annotation.tsv
- ../../outputs/data_generation/test_pre_annotation.tsv
- ../../outputs/data_generation/to_predict_1.tsv

3. Go annotate pre_annotation files. When finished, save the file names as below.

pre_annotation_1.tsv -> post_annotation_1.tsv
val_pre_annotation.tsv -> val_post_annotation.tsv
test_pre_annotation.tsv -> test_post_annotation.tsv

4. Run post_process_annotation.py --round=1

Output files
- ../../outputs/data_generation/train_1.tsv
- ../../outputs/data_generation/val.tsv
- ../../outputs/data_generation/test.tsv
- ../../outputs/kg/1/kg.txt
- ../../outputs/kg/1/entities.txt
- ../../outputs/kg/1/relations.txt

5. Train the entailment model.

Output files
- ../../outputs/data_generation/predicted_1.tsv

6. Generate round 2 pre annotation.
python generate_pre_annotation.py --round=2
