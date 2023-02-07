#!/bin/bash

N_START=1
N_RUNS=100

sbatch _al_certain_pos_job.sh $N_START $N_RUNS
sbatch _al_stratified_job.sh $N_START $N_RUNS
sbatch _al_uncertain_job.sh $N_START $N_RUNS
sbatch _al_random_job.sh $N_START $N_RUNS
