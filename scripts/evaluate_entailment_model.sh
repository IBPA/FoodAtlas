#!/bin/bash

cd ..

python -m food_atlas.entailment.evaluate \
    outputs/data_generation/test.tsv \
    biobert \
    outputs/round_1/entailment_model \
    --metric prec \
    --random-seed 42

cd scripts
