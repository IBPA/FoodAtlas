
![FoodAtlas Logo](./figures/foodatlas_logo_black.png)


# FoodAtlas ([foodatlas.ai](https://www.foodatlas.ai/))

FoodAtlas is a semi-automated framework for extracting food-chemical relationships from scientific literature using deep learning-based language models. The framework constructs a quality-controlled FoodAtlas knowledge graph regarding food composition.

We are actively working on developing the FoodAtlas website where you can access all of our data (coming soon in April, 2023; pending review). Please check back later for updates.

![Figure 1](./figures/Figure1.png)


## Prerequisites

This code has been tested with
* Python 3.8

To prevent dependency problems, please use either virtualenv...
```
# Activate Python virtualenv
python3 -mvenv env
source ./env/bin/activate

# Dectivate Python virtualenv
deactivate
```
or conda...
```
# Activate Conda environment
conda create -n mvenv python

# Deactivate Conda environment
conda deactivate
```

In your environment, please install python packages.
```
pip install -r requirement.txt
```

## Running

In order to reproduce the results obtained in our work, please follow the detailed steps below.

### Step 1. Generate the PH pairs and train/val/test set
Please follow the instructions in section 'Step 1' of the [README](./food_atlas/data_processing/README.md) file.

### Step 2. Train the entailmnet model
Please follow the instructions in the [README](./food_atlas/entailment/README.md) file.

### Step 3. Generate the FoodAtlas KG
Please follow the instructions in section 'Step 3' of the [README](./food_atlas/data_processing/README.md) file.

### Step 4. Perform knowledge graph completion
Please follow the instructions in the [README](./food_atlas/kgc/README.md) file.

## Authors

* **Jason Youn** @ [https://github.com/jasonyoun](https://github.com/jasonyoun)
* **Fangzhou Li** @ [https://github.com/fangzhouli](https://github.com/fangzhouli)

## Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## Citation

Citation will be updated later.

## License

This project is licensed under the Apache-2.0 License. Please see the <code>[LICENSE](./LICENSE)</code> file for details.

## Acknowledgments

Acknowledgments will be updated later.
