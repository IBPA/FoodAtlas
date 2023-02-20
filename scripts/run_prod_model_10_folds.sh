#!/bin/bash

sbatch _prod_model_10_folds_job.sh 0 12
sbatch _prod_model_10_folds_job.sh 13 25
sbatch _prod_model_10_folds_job.sh 26 37
sbatch _prod_model_10_folds_job.sh 38 49
