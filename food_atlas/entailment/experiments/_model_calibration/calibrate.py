# from sklearn.metrics import precision_score, confusion_matrix
# import matplotlib
import os

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import (
    calibration_curve,
    CalibrationDisplay,
    _SigmoidCalibration,
)
import seaborn as sns
import matplotlib.pyplot as plt

from ... import load_model, get_food_atlas_data_loader, evaluate
from ...utils import get_all_metrics


if __name__ == '__main__':
    PATH_OUTPUT = f"{os.path.abspath(os.path.dirname(__file__))}"\
        "/outputs/calibration"

    model = load_model('biobert')
    model, tokenizer = load_model('biobert')
    model.load_state_dict(
        torch.load('outputs/entailment_model/1/seed_4/model_state.pt'))

    data_loader_val = get_food_atlas_data_loader(
        'outputs/data_generation/val.tsv',
        tokenizer=tokenizer,
        train=True,
        batch_size=16,  # Does not matter for evaluation.
        shuffle=False)

    y_true, y_pred, y_score, loss = evaluate(
        model,
        data_loader_val,
        device='cuda')

    # Plot bin counts.
    sns.histplot(
        data=y_score,
        bins=10,
    )
    plt.savefig(f"{PATH_OUTPUT}/bin_counts.png")
    plt.close()

    # Before calibration.
    CalibrationDisplay.from_predictions(
        y_true, y_score, n_bins=10
    )
    plt.savefig(
        f"{PATH_OUTPUT}/reliability_diagram_pre.png", bbox_inches='tight')
    plt.close()

    # After calibration.
    calibrator = _SigmoidCalibration()
    calibrator.fit(y_score, y_true)
    y_cali = calibrator.predict(y_score)
    CalibrationDisplay.from_predictions(
        y_true, y_cali, n_bins=10
    )
    plt.savefig(
        f"{PATH_OUTPUT}/reliability_diagram_post.png", bbox_inches='tight')
    plt.close()
