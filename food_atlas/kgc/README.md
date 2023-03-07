## Step 5. Perform knowledge graph completion (KGC)

We will not perform KGC using different link prediction models.

```
cd pykeen
```

### 5a. Generate data

We first generate data that will be used for comparing the KGC results.
```
./generate_data.sh
```

### 5b. Run hyperparameter optimization for different models and datasets

We need to optimize the hyperparameters for multiple models and multiple datasets. The following script runs the HPO in a sequential manner, so you may wish to change the script based on your own needs.
```
./run_hpo.sh
```

### 5c. Run the best model on the test set and generate statistics

Using the best hyperparameter found above for each dataset and model pair, we get the statistics on the test set.
```
./run_test_using_best_models.sh
```
