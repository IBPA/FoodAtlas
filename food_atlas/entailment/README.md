## Step 2. Entailment model training

This section provides detailed explanation how entailment models are trained and inferred.

### 2a. Configure the output path

**Before running, you must make sure to specify output path with enough disk space (>= 2 TB)**. This can be done by changing `PATH_OUTPUT_ROOT` parameter in `FoodAtlas/scripts/_run_round_job.sh` file. For each active learning round (i.e., 4000 rounds = 4 AL * 100 runs/AL * 10 rounds/run), this output folder will be dumped with the following files:
- the grid search result
- the best model weights
- the evaluation statistics for the best model
- and the predictions for all unlabelled data using the best model

The production models are also going to be dumped in this folder, which is an ensembled model of 100 language models.

Output Directory File Structure:
```
{PATH_OUTPUT_ROOT}/entailment_model
├── certain_pos
│   ├── run_1
│   │   ├── round_1
│   │   │   ├── eval_best_model
│   │   │   │   ├── eval_results.csv
│   │   │   │   └── seed_20001
│   │   │   │       ├── model_state.pt
│   │   │   │       └── result.pkl
│   │   │   └── grid_search
│   │   │       ├── grid_search_result.pkl
│   │   │       └── grid_search_result_summary.csv
│   │   ├── round_2
|   |   |   └── ...
|   |   ...
│   ├── run_2
│   │   └── ...
│   ...
├── uncertain
│   └── ..
├── random
│   └── ..
├── stratified
│   └── ..
└── prod (See 2c)
    └── ..
```

### 2b. Simulate active learning

Once the data are ready from [Step 1](../data_processing/README.md), and the output path is configured, we can use them to train/validate/test each active learning strategies.


Once ready, you can run the following command to start the simulation:

```console
cd FoodAtlas/scripts
./run_all_active_learning.sh
```

Other parameters:
- To modify SLURM parameters, go to `FoodAtlas/scripts/_al_*_job.sh` files.
- To modify the number of active learning runs to reduce disk space usage, go to `FoodAtlas/scripts/_run_round_job.sh` file.

### 2c. Production model ensembling

Output Directory File Structire:

```console
prod
├── ensemble
│   ├── 0
│   │   └── seed_0
│   │       ├── model_state.pt
│   │       ├── predicted.tsv
│   │       └── result_train.pkl
│   ├── 1
│   │   └── seed_1
│   │       └── ...
│   ...
└── grid_search
    ├── fold_0
    │   └── grid_search
    │       ├── grid_search_result.pkl
    │       └── grid_search_result_summary.csv
    ├── fold_1
    │   └── ...
    ...
    └── fold_9
        └── ...
```

Once we finished the simulation, we can merge train/val/test datasets all together to train the production model.

#### 2c1. Production model hyperparameter tuning

First, we use 10-fold cross validation to find the best hyperparameters for the production model.

```console
./run_prod_model_tuning.sh
```

The outputs of the grid search will be dumped in `prod/grid_search` folder.

#### 2c2. Production model training

Then, we will use the best hyperparameters to train 100 production model with different random seeds.

```console
./run_prod_model_ensemble.sh
```

The `model_state.pt` and `result_train.pkl` of the 100 production models will be dumped in the corresponding folders in `prod/ensemble`.

#### 2c3. Production model ensemble prediction

Finally, we will ensemble the 100 production models to get predictions for all unlabeled data.

```console
./run_prod_model_ensemble_pred.sh
```

The `predicted.tsv` of the 100 production models will be dumped in the corresponding folders in `prod/ensemble`.


