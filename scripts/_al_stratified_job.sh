#!/bin/bash

#SBATCH --job-name=FA_grid_search
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/IBPA/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/IBPA/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=20-00:00:00

N_START=$1
N_RUNS=$2

for (( run=$N_START; run<=$N_RUNS; run++ ))
do
    for round in {1..10}
    do
        ./_run_round_job.sh stratified $run $round
    done
done
