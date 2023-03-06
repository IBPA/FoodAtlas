import argparse
import os
import sys

import numpy as np
import pandas as pd
from pykeen.datasets.base import PathDataset
from pykeen.evaluation.classification_evaluator import ClassificationMetricResults
from pykeen.pipeline.api import pipeline_from_path
from pykeen.predict import predict_triples
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from utils import determine_threshold


TRAIN_FILENAME = "train.txt"
VAL_FILENAME = "val.txt"
VAL_CORRUPTED_FILENAME = "val_corrupted.txt"
VAL_CORRUPTED_NO_LABEL_FILENAME = "val_corrupted_no_label.txt"
TEST_FILENAME = "test.txt"
TEST_CORRUPTED_FILENAME = "test_corrupted.txt"
TEST_CORRUPTED_NO_LABEL_FILENAME = "test_corrupted_no_label.txt"
BEST_MODEL_METRICS_FILTERED_FILENAME = 'best_model_metrics_filtered.tsv'
BEST_MODEL_METRICS_UNFILTERED_FILENAME = 'best_model_metrics_unfiltered.tsv'


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="Dataset directory.",
    )

    parser.add_argument(
        "--val_test_dir",
        type=str,
        required=True,
        help="Directory containing val/test set.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory.",
    )

    parser.add_argument(
        "--num_replications",
        type=int,
        required=True,
        help="Number of replications for statistics.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    return parser.parse_args()


def _generate_report(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'precision_score': precision_score(y_true, y_pred),
        'recall_score': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'accuracy_score': accuracy_score(y_true, y_pred),
        'average_precision_score': average_precision_score(y_true, y_score),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'roc_auc_score': roc_auc_score(y_true, y_score),
    }


def _get_triple_str(row, prefix=''):
    return f"({row['head'+prefix]}, {row['relation'+prefix]}, {row['tail'+prefix]})"


def main():
    args = parse_argument()

    # Get a dataset
    dataset = PathDataset(
        training_path=os.path.join(args.input_kg_dir, TRAIN_FILENAME),
        validation_path=os.path.join(args.val_test_dir, VAL_CORRUPTED_NO_LABEL_FILENAME),
        testing_path=os.path.join(args.val_test_dir, TEST_CORRUPTED_NO_LABEL_FILENAME),
        create_inverse_triples=True,
    )

    best_pipeline_config_filepath = os.path.join(
        args.input_dir, 'results', 'best_pipeline', 'pipeline_config.json')

    # build a triple lookup because pykeen doesn't allow
    # passing labels to the dataset class
    df_val = pd.read_csv(
        os.path.join(args.val_test_dir, VAL_CORRUPTED_FILENAME),
        sep='\t',
        names=['head', 'relation', 'tail', 'label'],
    )
    df_val['triple'] = df_val.apply(lambda row: _get_triple_str(row), axis=1)
    val_triple_label_lookup = dict(zip(
        df_val['triple'].tolist(),
        df_val['label'].tolist(),
    ))

    df_test = pd.read_csv(
        os.path.join(args.val_test_dir, TEST_CORRUPTED_FILENAME),
        sep='\t',
        names=['head', 'relation', 'tail', 'label'],
    )
    df_test['triple'] = df_test.apply(lambda row: _get_triple_str(row), axis=1)
    test_triple_label_lookup = dict(zip(
        df_test['triple'].tolist(),
        df_test['label'].tolist(),
    ))

    filtered = []
    unfiltered = []
    for num_replicate in range(args.num_replications):
        print(f'\n\nRunning replicate {num_replicate+1}/{args.num_replications}...')

        pipeline_result = pipeline_from_path(
            best_pipeline_config_filepath,
            dataset=dataset,
        )

        # find threshold using validation set
        val_predictions = predict_triples(
            model=pipeline_result.model,
            triples=dataset.validation.mapped_triples,
        )

        df = val_predictions.process(factory=pipeline_result.training).df
        df['triple'] = df.apply(lambda row: _get_triple_str(row, prefix='_label'), axis=1)
        df['label'] = df['triple'].apply(lambda x: val_triple_label_lookup[x])
        threshold = determine_threshold(df['score'].tolist(), df['label'].tolist())
        print(f'Threshold: {threshold}')

        # make predictions using test set
        test_predictions = predict_triples(
            model=pipeline_result.model,
            triples=dataset.testing.mapped_triples,
        )

        df = test_predictions.process(factory=pipeline_result.training).df
        df['triple'] = df.apply(lambda row: _get_triple_str(row, prefix='_label'), axis=1)
        df['label'] = df['triple'].apply(lambda x: test_triple_label_lookup[x])

        y_true = df['label'].tolist()
        y_pred = [x >= threshold for x in df['score'].tolist()]
        y_score = df['score'].tolist()

        filtered.append(_generate_report(y_true, y_pred, y_score))

        # get unfiltered now using the test set that's not dropping the unseen triples
        filtered_triples = df['triple'].tolist()
        all_triples = df_test['triple'].tolist()

        y_true_unfiltered = [
            test_triple_label_lookup[x] for x in all_triples if x not in filtered_triples]
        y_pred_unfiltered = [1-x for x in y_true_unfiltered]
        y_score_unfiltered = [min(y_score) if x == 0 else max(y_score) for x in y_pred_unfiltered]

        y_true += y_true_unfiltered
        y_pred += y_pred_unfiltered
        y_score += y_score_unfiltered

        unfiltered.append(_generate_report(y_true, y_pred, y_score))

    df_metrics_filtered = pd.DataFrame(filtered)
    df_metrics_filtered.to_csv(
        os.path.join(args.output_dir, BEST_MODEL_METRICS_FILTERED_FILENAME),
        sep='\t',
        index=False,
    )

    df_metrics_unfiltered = pd.DataFrame(unfiltered)
    df_metrics_unfiltered.to_csv(
        os.path.join(args.output_dir, BEST_MODEL_METRICS_UNFILTERED_FILENAME),
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    main()
