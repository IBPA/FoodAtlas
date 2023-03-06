#!/bin/bash

# You can change N_RUNS to be smaller if you do not want 100 active learning runs.
N_RUNS=100

N_START=1

sbatch _al_certain_pos_job.sh $N_START $N_RUNS
sbatch _al_stratified_job.sh $N_START $N_RUNS
sbatch _al_uncertain_job.sh $N_START $N_RUNS
sbatch _al_random_job.sh $N_START $N_RUNS
