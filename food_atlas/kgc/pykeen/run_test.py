import argparse
import json
import os
import sys
from time import time
from datetime import datetime

import numpy as np
import pandas as pd
from pykeen.datasets.base import PathDataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline.api import pipeline_from_path, pipeline_from_config
from pykeen.pipeline.plot_utils import plot_er
from pykeen.predict import predict_triples
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    r2_score, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.calibration import CalibrationDisplay
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR
from pandarallel import pandarallel
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../data_processing/')
from utils.utils import determine_threshold, generate_report  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402


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
        "--train_dir",
        type=str,
        required=True,
        help="Train dir.",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Validation dir.",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Test dir.",
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

    parser.add_argument(
        "--generate_test_stats",
        action='store_true',
        help="Set if generating test stats.",
    )

    parser.add_argument(
        "--is_production",
        action='store_true',
        help="Set if this is a production model.",
    )

    args = parser.parse_args()

    if args.is_production:
        assert args.generate_test_stats is False

    return args


def _get_triple_str(row, prefix=''):
    return f"({row['head'+prefix]}, {row['relation'+prefix]}, {row['tail'+prefix]})"


def get_diagram_data(label, score, n_bins=5):
    if n_bins == 5:
        cutoff = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    elif n_bins == 10:
        cutoff = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        raise ValueError

    x = []
    y = []
    for i in range(len(cutoff)-1):
        lower_cutoff = cutoff[i]
        upper_cutoff = cutoff[i+1]

        if i < (n_bins-1):
            index = [i for i, x in enumerate(score) if x >= lower_cutoff and x < upper_cutoff]
        else:
            index = [i for i, x in enumerate(score) if x >= lower_cutoff]

        label_selected = [label[x] for x in index]
        x.append((lower_cutoff+upper_cutoff)/2)
        y.append(label_selected.count(1)/len(label_selected))

    return x, y


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    # Get a dataset
    if args.is_production:
        testing_path = os.path.join(args.test_dir, TEST_FILENAME)
    else:
        testing_path = os.path.join(args.test_dir, TEST_CORRUPTED_NO_LABEL_FILENAME)

    dataset = PathDataset(
        training_path=os.path.join(args.train_dir, TRAIN_FILENAME),
        validation_path=os.path.join(args.val_dir, VAL_CORRUPTED_NO_LABEL_FILENAME),
        testing_path=testing_path,
        create_inverse_triples=True,
    )

    if not args.is_production:
        rank_based_evalution_dataset = PathDataset(
            training_path=os.path.join(args.train_dir, TRAIN_FILENAME),
            validation_path=os.path.join(args.val_dir, VAL_FILENAME),
            testing_path=os.path.join(args.val_dir, TEST_FILENAME),
            create_inverse_triples=True,
        )

    best_pipeline_config_filepath = os.path.join(
        args.input_dir, 'results', 'best_pipeline', 'pipeline_config.json')

    # build a triple lookup because pykeen doesn't allow
    # passing labels to the dataset class
    df_val = pd.read_csv(
        os.path.join(args.val_dir, VAL_CORRUPTED_FILENAME),
        sep='\t', names=['head', 'relation', 'tail', 'label'],
    )
    df_val['triple'] = df_val.apply(lambda row: _get_triple_str(row), axis=1)
    val_triple_label_lookup = dict(zip(
        df_val['triple'].tolist(),
        df_val['label'].tolist(),
    ))

    if not args.is_production:
        df_test = pd.read_csv(
            os.path.join(args.test_dir, TEST_CORRUPTED_FILENAME),
            sep='\t', names=['head', 'relation', 'tail', 'label'],
        )
        df_test['triple'] = df_test.apply(lambda row: _get_triple_str(row), axis=1)
        test_triple_label_lookup = dict(zip(
            df_test['triple'].tolist(),
            df_test['label'].tolist(),
        ))
    else:
        df_test = pd.read_csv(
            os.path.join(args.test_dir, TEST_FILENAME),
            sep='\t', names=['head', 'relation', 'tail'],
        )
        df_test['triple'] = df_test.apply(lambda row: _get_triple_str(row), axis=1)

    filtered = []
    unfiltered = []
    labels = []
    scores = []
    results = []
    for num_replicate in range(args.num_replications):
        print(f'\n\nRunning replicate {num_replicate+1}/{args.num_replications}...')
        scaler = MinMaxScaler()
        ir = IR(out_of_bounds='clip')

        print('Running pipeline...')
        pipeline_result = pipeline_from_path(
            best_pipeline_config_filepath,
            dataset=dataset,
        )

        if not args.is_production:
            print('Running RankBasedEvaluator...')
            evaluator = RankBasedEvaluator(filtered=True)
            r = evaluator.evaluate(
                model=pipeline_result.model,
                mapped_triples=rank_based_evalution_dataset.testing.mapped_triples,
                additional_filter_triples=[
                    rank_based_evalution_dataset.training.mapped_triples,
                    rank_based_evalution_dataset.validation.mapped_triples,
                ],
            )
            results.append(r.to_flat_dict())

        # find threshold using validation set
        print('Running prediction on validation set...')
        val_predictions = predict_triples(
            model=pipeline_result.model,
            triples=dataset.validation.mapped_triples,
        )

        df = val_predictions.process(factory=pipeline_result.training).df
        df['triple'] = df.apply(lambda row: _get_triple_str(row, prefix='_label'), axis=1)
        df['label'] = df['triple'].apply(lambda x: val_triple_label_lookup[x])
        ir.fit(df['score'].to_numpy(), df['label'].to_numpy())
        df['score'] = ir.transform(df['score'].to_numpy())
        scaler.fit(df[['score']])
        df[['score']] = scaler.transform(df[['score']])

        threshold = determine_threshold(df['score'].tolist(), df['label'].tolist())
        print(f'Threshold: {threshold}')

        # make predictions using test set
        print('Running prediction on test set...')
        test_predictions = predict_triples(
            model=pipeline_result.model,
            triples=dataset.testing.mapped_triples,
        )

        df = test_predictions.process(factory=pipeline_result.training).df
        df['triple'] = df.apply(lambda row: _get_triple_str(row, prefix='_label'), axis=1)
        df['score'] = ir.predict(df['score'].to_numpy())
        df[['score']] = scaler.transform(df[['score']])

        if args.is_production:
            test_triple_score_dict = dict(zip(df['triple'].tolist(), df['score'].tolist()))
            df_test[f'prob_run_{num_replicate}'] = df_test['triple'].parallel_apply(
                lambda x: test_triple_score_dict[x] if x in test_triple_score_dict else np.nan
            )
            df_test[f'pred_run_{num_replicate}'] = \
                df_test[f'prob_run_{num_replicate}'].parallel_apply(
                    lambda x: False if np.isnan(x) else x > threshold
                )

        else:
            df['label'] = df['triple'].apply(lambda x: test_triple_label_lookup[x])

            y_true = df['label'].tolist()
            y_pred = [x >= threshold for x in df['score'].tolist()]
            y_score = df['score'].tolist()

            filtered.append(generate_report(y_true, y_pred, y_score))

            # get unfiltered now using the test set that's not dropping the unseen triples
            filtered_triples = df['triple'].tolist()
            all_triples = df_test['triple'].tolist()

            y_true_unfiltered = [
                test_triple_label_lookup[x] for x in all_triples if x not in filtered_triples]
            y_pred_unfiltered = [0 for _ in y_true_unfiltered]
            y_score_unfiltered = [min(y_score) for _ in y_pred_unfiltered]

            y_true += y_true_unfiltered
            y_pred += y_pred_unfiltered
            y_score += y_score_unfiltered

            unfiltered.append(generate_report(y_true, y_pred, y_score))

            labels.append(y_true)
            scores.append(y_score)

    if args.is_production:
        def _calculate_mean_std(row):
            probs = []
            prob_names = [x for x in row.index.values.tolist() if x.startswith('prob_run_')]
            for x in prob_names:
                probs.append(row[x])
            return [np.mean(probs), np.std(probs)]

        df_test[['prob_mean', 'prob_std']] = df_test.parallel_apply(
            lambda row: _calculate_mean_std(row), axis=1, result_type='expand')
        df_test.drop('triple', axis=1, inplace=True)
        df_test.sort_values('prob_mean', ascending=False, inplace=True)

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        df_test.to_csv(
            os.path.join(args.output_dir, f'hypotheses_{dt_string}.tsv'),
            sep='\t',
            index=False,
        )

    if args.generate_test_stats:
        # rank based metrics
        df_rank_metrics = pd.DataFrame(results)
        df_rank_metrics.to_csv(
            os.path.join(args.output_dir, 'rank_based_metrics.tsv'),
            sep='\t', index=False,
        )

        # plot calibration plot (individual)
        assert len(labels) == len(scores)

        plt.figure()
        for i in range(len(labels)):
            x, y = get_diagram_data(labels[i], scores[i])
            plt.plot(x, y)
        plt.savefig(os.path.join(args.output_dir, 'calibration_plot.png'))
        plt.savefig(os.path.join(args.output_dir, 'calibration_plot.svg'))

        # plot calibration plot (flattened)
        flattened_labels = [y for x in labels for y in x]
        flattened_scores = [y for x in scores for y in x]
        x, y = get_diagram_data(flattened_labels, flattened_scores)
        plt.figure()
        plt.plot(x, y)
        plt.text(0.2, 0.2, f'R2 score: {r2_score(x, y)}')
        plt.savefig(os.path.join(args.output_dir, 'calibration_plot_flat.png'))
        plt.savefig(os.path.join(args.output_dir, 'calibration_plot_flat.svg'))

        # plot PR curve (individual)
        plt.figure()
        for i in range(len(labels)):
            precision, recall, _ = precision_recall_curve(labels[i], scores[i])
            plt.plot(recall, precision)
        plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'))
        plt.savefig(os.path.join(args.output_dir, 'pr_curve.svg'))

        # plot PR curve (flattened)
        plt.figure()
        precision, recall, _ = precision_recall_curve(flattened_labels, flattened_scores)
        plt.plot(recall, precision)
        plt.text(0.2, 0.2, f'AUCPR: {average_precision_score(flattened_labels, flattened_scores)}')
        plt.savefig(os.path.join(args.output_dir, 'pr_curve_flat.png'))
        plt.savefig(os.path.join(args.output_dir, 'pr_curve_flat.svg'))

        # plot ROC curve (individual)
        plt.figure()
        for i in range(len(labels)):
            fpr, tpr, _ = roc_curve(labels[i], scores[i])
            plt.plot(fpr, tpr)
        plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
        plt.savefig(os.path.join(args.output_dir, 'roc_curve.svg'))

        # plot ROC curve (flattened)
        plt.figure()
        fpr, tpr, _ = roc_curve(flattened_labels, flattened_scores)
        plt.plot(fpr, tpr)
        plt.text(0.2, 0.2, f'AUROC: {roc_auc_score(flattened_labels, flattened_scores)}')
        plt.savefig(os.path.join(args.output_dir, 'roc_curve_flat.png'))
        plt.savefig(os.path.join(args.output_dir, 'roc_curve_flat.svg'))

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
