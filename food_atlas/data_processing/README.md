## Step 1. Generate PH pairs and train/val/test set

This section shows a step-by-step instructions on how to generate the PH pairs, perform annotation, and generate post-process the annotations to generate train/val/test set.

### 1a. Generate annotation data

We first generate the PH pairs. Make sure to replace the `--cache_dir` to some temporary directory in your local filesystem.
```
python query_and_generate_ph_pairs.py \
    --input_type=query_string \
    --query_filepath=../../data/FooDB/foodb_queries.txt \
    --allowed_ncbi_taxids_filepath=../../data/FoodAtlas/allowed_ncbi_taxids.tsv \
    --cache_dir=/home/jasonyoun/Temp
```

Output files are as follows:
```
- ../../outputs/data_processing/query_results.txt
- ../../outputs/data_processing/ph_pairs_{timestamp}.txt
```

### 1b. Generate pre-annotation data

Using the PH pairs generated above, we randomly generate sample train/val/test set, which is ready for annotation.
```
python generate_pre_annotation.py \
    --train_pre_annotation_filepath=../../outputs/data_processing/train_pool_pre_annotation.tsv
```

Output files are as follows:
```
- ../../outputs/data_processing/train_pool_pre_annotation.tsv
- ../../outputs/data_processing/val_pre_annotation.tsv
- ../../outputs/data_processing/test_pre_annotation.tsv
- ../../outputs/data_processing/to_predict.tsv
```

### 1c. Annotation time!

We used [Label Studio](https://labelstud.io/) deployed on [Heroku](https://www.heroku.com/) to annnotate the PH pairs. Once finished with annotation, export the annotation files as a .tsv file with the name format specified below for each dataset.
```
# Input (train)
../../outputs/data_processing/train_pool_pre_annotation.tsv
# Output
../../outputs/data_processing/train_pool_post_annotation.tsv

# Input (val)
../../outputs/data_processing/val_pre_annotation.tsv
# Output
../../outputs/data_processing/val_post_annotation.tsv

# Input (test)
../../outputs/data_processing/test_pre_annotation.tsv
# Output
../../outputs/data_processing/test_post_annotation.tsv
```

### 1d. Post process annotation

We now need to post-process the annotation to generate a clean version of train/val/test set.
```
python post_process_annotation.py \
    --train_post_annotation_filepath=../../outputs/data_processing/train_pool_post_annotation.tsv \
    --train_filepath=../../outputs/data_processing/train_pool.tsv
```

Output files are as follows:
```
- ../../outputs/data_processing/train_pool.tsv
- ../../outputs/data_processing/val.tsv
- ../../outputs/data_processing/test.tsv
```

### 1e. Generate data for deployment entailment model
We need to do hyperparameter optimization for the deployment (final) entailment model. Run the following Python script to generate the necessary files.
```
python generate_folds.py \
    --input_train_filepath=../../outputs/data_processing/train_pool.tsv \
    --input_val_filepath=../../outputs/data_processing/val.tsv \
    --input_test_filepath=../../outputs/data_processing/test.tsv \
    --output_dir=../../outputs/data_processing/folds_for_prod_model
```

### 1f. (Optional) Generate more data

Following the above steps finished the data generation process (PH pairs and train/val/test set). In this work, we generated additional PH pairs.

We generated more queries using the food-chemical pairs extracted from each external DB as follows.
```
# Frida
python generate_food_chem_queries.py \
    --input_filepath=../../data/Frida/frida.tsv \
    --output_filepath=../../data/Frida/frida_queries.txt

# Phenol-Explorer
python generate_food_chem_queries.py \
    --input_filepath=../../data/Phenol-Explorer/phenol_explorer.tsv \
    --output_filepath=../../data/Phenol-Explorer/phenol_explorer_queries.txt
```

We then generated PH pairs using the queries generated above.
```
# Frida
python query_and_generate_ph_pairs.py \
    --input_type=query_string \
    --query_filepath=../../data/Frida/frida_queries.txt \
    --allowed_ncbi_taxids_filepath=../../data/FoodAtlas/allowed_ncbi_taxids.tsv \
    --cache_dir=/home/jasonyoun/Temp

# Phenol-Explorer
python query_and_generate_ph_pairs.py \
    --input_type=query_string \
    --query_filepath=../../data/Phenol-Explorer/phenol_explorer_queries.txt \
    --allowed_ncbi_taxids_filepath=../../data/FoodAtlas/allowed_ncbi_taxids.tsv \
    --cache_dir=/home/jasonyoun/Temp
```

The [LitSense](https://www.ncbi.nlm.nih.gov/research/litsense/) API is limited to 100 results for a given query. We collaborated with the LitSense team to internally generate bigger query results (maximum 50,000 results for a given query). We then used these results to generate the PH pairs.
```
python query_and_generate_ph_pairs.py \
    --input_type=query_results \
    --query_filepath=../../data/FoodAtlas/litsense_query/queries_output/*.json \
    --allowed_ncbi_taxids_filepath=../../data/FoodAtlas/allowed_ncbi_taxids.tsv \
    --cache_dir=/home/jasonyoun/Temp
```


## Step 2. Train the entailmnet model

We now need to train the entailment model using the train/val/test set generated above. Please refer to its own [README](../entailment/README.md) file. Once finished, come back and follow the remaining steps.


## Step 3. Generate the Knowledge Graph (KG).
Run the script as below to generate the KG. The script was ran on a PC with 12 cores and 64 GB of RAM. Depending on your computer, you may want to adjust the `--nb_workers` argument to fit your needs. Please refer to the script for detailed steps.

```
./kg.sh
```
