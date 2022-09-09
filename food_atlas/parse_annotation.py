# -*- coding: utf-8 -*-
"""One line summary.

More detailed description.

Attributes:
    attr1 (type): Description of attr1.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * TODOs

"""
import os

import pandas as pd


def label_ph_pair(
        annots: list) -> str:
    """Label a PH pair based on annotation answers.

    Args:
        annots (list): List of answers.

    Returns:
        str: Label of the PH pair.

    """
    if len(annots) != 2:
        raise ValueError('Annotations must be a list of length 2.')

    if annots[0] == annots[1]:
        return annots[0]
    else:
        return 'Skip'


def parse_annotation(path_annotation):
    """
    """
    annots = pd.read_csv(path_annotation)
    annots_grouped = annots.groupby('id')
    data = annots_grouped[
        ['head', 'tail', 'relation', 'premise', 'hypothesis_string']
    ].apply(lambda group: group.iloc[0])
    data['annotations'] = annots_grouped['answer'].apply(
        lambda group: group.tolist()
    )
    data['label'] = data['annotations'].apply(label_ph_pair)
    data = data[data['label'] != 'Skip']
    data = data.reset_index().rename({'id': 'label_studio_id'}, axis=1)

    return data


if __name__ == '__main__':
    PATH_DATA = (
        f"{os.path.abspath(os.path.dirname(__file__))}/../data/"
        "annotations/project-1-at-2022-09-08-22-55-c3f8bdbb.csv")

    data = parse_annotation(PATH_DATA)
    print(data)
