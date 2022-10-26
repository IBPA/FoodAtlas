#!/bin/bash

#SBATCH --job-name=FA_grid_search
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/home/lfz/git/FoodAtlas/logs/%j.out
#SBATCH --error=/home/lfz/git/FoodAtlas/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=10-00:00:00

for run in {3..100}
do
    for round in {1..10}
    do
        ./run_round.sh stratified $run $round
    done
done
