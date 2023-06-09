
![FoodAtlas Logo](./figures/foodatlas_logo_black.png)

# FoodAtlas ([foodatlas.ai](https://www.foodatlas.ai/))

## Overview

### Introduction

We present FoodAtlas, a quality-controlled food-chemical knowledge graph constructed by a semi-automated framework to extract information from scientific literature using deep learning-based models (Figure 1). This framework, Lit2KG, utilized active learning achieving 38% ...

<!-- for extracting food-chemical relationships from scientific literature using deep learning-based language models, and in turn, constructing a quality-controlled FoodAtlas knowledge graph regarding food composition (Figure 1). We tested 4 different active learning strategies and found that selecting samples that maximize the likelihood leads to discovering new knowledge 38.2% faster than the random sampling baseline. The entailment models achieved an average of 0.82 precision, 0,84 recall, and 0.83 F1 score in extracting the food-chemical relations from the premise in the final round of active learning, with the predicted probabilities being highly correlated to the ground-truth annotations ($R^2$ = 0.94). The FoodAtlas KG with 291,775 triples integrates knowledge from the entailment model pipeline, link prediction, multiple external databases, as well as taxonomical and ontological relationships of foods and chemicals, respectively. -->

We are actively working on developing the [*FoodAtlas website*](https://www.foodatlas.ai/) where you can access all of our data.

*Figure 1. Overview of the FoodAtlas framework.*
![Figure 1](./figures/Figure1.png)

* **a** Summary of our Lit2KG architecture. We query scientific literature using LitSense, an API provided by NCBI for the sentence-level search of biomedical articles from PubMed and PubMed Central, using food names (e.g., cocoa) collected from multiple databases. We use the entities tagged in each sentence (premise) returned by the LitSense API (e.g., … cocoa[SPECIES] is a good source of (-)-epicatechin[CHEMICAL] …) to generate hypotheses triple like (cocoa, contains, (-)-epicatechin), which we refer to as premise-hypothesis (PH) pairs.
* For the first round of active learning (AL), a subset of the PH pairs is sampled randomly and annotated by experts, which are in turn used to train the entailment model that classifies which one of these PH pairs is positive (the premise entails the hypothesis).
* For the subsequent rounds of AL, we utilize the entailment model predictions to choose the next batch of PH pairs to sample and annotate.
* We construct the FoodAtlas knowledge graph (KG) using annotated and unannotated PH pairs, food-chemical composition information from external databases, as well as taxonomical and ontological relationships of food and chemical entities, respectively. The FoodAtlas KG keeps evidence of where each triple is extracted from to provide reproducible data.
* Finally, we perform link prediciton on the FoodAtlas KG to infer new food-chemcal relationships.

### Results

*Figure 2. Prediction performance of the entailment model.*
![Figure 2](./figures/Figure2.png)

* **a**: Precision, recall, and F1 score of the entailment models trained using the 4 different active learning (AL) strategies for rounds 1 through 10. On left, the line plot shows the mean value of each AL strategy and the error lines denote the standard deviation of the 100 random seeds. On right, the box represents the interquartile range, the middle line represents the median, the whisker line extends from minimum to maximum values, and the diamond represents outliers.
* **b**: Comparison of the new knowledge discovery rate compared between the 4 AL strategies. The plot shows how early on in the AL round the 1,899 positive triples within the simulated training pool of 4,120 positives are discovered. The error line shows the standard deviation of the 100 random seeds.
* **c**: Calibration plot showing a high correlation between the probability assigned by the entailment model and the ground-truth annotations on the test set. (R2 = 0.94).
* **d, e**: The precision-recall (PR) and receiver operating characteristic (ROC) curves of the entailment model predictions compared to the ground-truth annotations in the test set at the final round (r = 10) averaged over all 400 runs with a different random seed (100 runs for each of the 4 AL strategies).

*Figure 3. Summary of the FoodAtlas Knowledge Graph.*
![Figure 3](./figures/Figure3.png)

* **a**: Schema of the FoodAtlas knowledge graph (KG) consisting of three unique entity (node) types and 4 unique relation (edge)  types. *contains* encodes the food-chemical composition relations, *hasPart* encodes the food-food with part relations, *isA* encodes the chemical ontological relations using the MeSH tree, and *hasChild* encodes the taxonomical relations using the NCBI taxonomy.
* **b**: Number of triples per data source in the FoodAtlas KG with different source quality types.
* **c**: Sankey graph showing the connections between quality, data source, and evidence. The thickness of the relations between the nodes represents the number of connections in the log scale.
* **d, e**: UpSet plot showing the number of unique triples for all data sources for all relation types (d) and all sources based on quality for only the contains triples (e). Each row in the plot corresponds to a source, and the bar chart on the left shows the size of each source. Each column corresponds to an intersection, where the filled-in cells denote which source is part of an intersection. The bar chart for each column denotes the size of intersections.
* **f, g**: Classification of foods and chemicals into two-level groups and subgroups.

*Figure 4. Prediction Performance of the link predictor.*
![Figure 4](./figures/Figure4.png)

* **a**: The summary
* **b**:
* **c**:
* **d**:

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

### Step 1. Download data
Download all input data first.
```
cd data
./download_and_process_data.sh
```

### Step 2. Generate the PH pairs and train/val/test set
Please follow the instructions in the [README](./food_atlas/data_processing/README.md) file.

### Step 3. Train the entailmnet model
Please follow the instructions in the [README](./food_atlas/entailment/README.md) file.

### Step 4. Generate the FoodAtlas KG
Run the script as below to generate the KG. The script was ran on a PC with 12 cores and 64 GB of RAM. Depending on your computer, you may want to adjust the `--nb_workers` argument to fit your needs. Please refer to the script for detailed steps.

```
cd ./food_atlas/data_processing
./kg.sh
```

### Step 5. Perform knowledge graph completion
Please follow the instructions in the [README](./food_atlas/kgc/README.md) file.

## Authors

* Jason Youn<sup>1,2,3</sup>
* Fangzhou Li<sup>1,2,3</sup>
* Gabriel Simmons<sup>1,2,3</sup>
* Ilias Tagkopoulos<sup>1,2,3</sup>

1. Department of Computer Science, the University of California at Davis
2. Genome Center, the University of California at Davis
3. USDA/NSF AI Institute for Next Generation Food Systems (AIFS)

## Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## Citation

Citation will be updated later.

## License

This project is licensed under the Apache-2.0 License. Please see the <code>[LICENSE](./LICENSE)</code> file for details.

## Acknowledgements
* Alexis Allot from National Center for Biotechnology Information (NCBI) for runninng LitSense queries internally.
* Kyle McKillop and Kai Blumberg from U.S. Department of Agriculture Agricultural Research Service (USDA ARS) for providing the FoodData Central (FDC) data internally.
* Anders Poulsen from Technical University of Denmark (DTU) for providing the Frida data internally.
* Navneet Rai and Adil Muhammad from the Tagkopoulos lab for annotating the PH pairs.

## Funding

* USDA-NIFA AI Institute for Next Generation Food Systems (AIFS), USDA-NIFA award number 2020-67021-32855.
