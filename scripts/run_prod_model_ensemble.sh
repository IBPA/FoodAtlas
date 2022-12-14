#!/bin/bash

sbatch _prod_model_ensemble_job.sh 0 33
sbatch _prod_model_ensemble_job.sh 34 66
sbatch _prod_model_ensemble_job.sh 67 99
