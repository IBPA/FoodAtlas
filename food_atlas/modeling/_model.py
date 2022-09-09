# -*- coding: utf-8 -*-
"""FoodAtlas NLI model loading module.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * TODOs

"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


def load_model(path_or_name):
    """
    """
    if path_or_name == 'biobert':
        model = AutoModelForSequenceClassification.from_pretrained(
            'dmis-lab/biobert-v1.1', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        tokenizer._pad_token_type_id = 1
    else:
        raise NotImplementedError(
            f"Model {path_or_name} not implemented."
        )

    return model, tokenizer
