#!/bin/bash

cd ..

python -m food_atlas.entailment.run_model \
    outputs/data_generation/train_1.tsv \
    outputs/data_generation/test.tsv \
    outputs/round_1/entailment_model \
    biobert \
    --metric prec \
    --random-seed 42

cd scripts
