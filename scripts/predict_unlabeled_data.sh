#!/bin/bash

cd ..

python -m food_atlas.entailment.predict \
    outputs/data_generation/to_test_1.tsv \
    biobert \
    outputs/round_1/entailment_model/best_prec/model_state.pt \
    outputs/data_generation/predicted_1.tsv

cd scripts
