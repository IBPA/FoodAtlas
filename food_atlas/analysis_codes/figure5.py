from ast import literal_eval
from collections import Counter
from glob import glob
import os
from pathlib import Path
import sys

sys.path.append('../data_processing/')

import seaborn as sns  # noqa: E402
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

KGC_DATA_DIR = '../../outputs/kgc/data'
RESULTS_ROOT_DIR = '../../outputs/kgc/pykeen'
CHATGPT_METRICS_FILEPATH = '../../outputs/kgc/chatgpt'
FILTERED_FILENAME = 'best_model_metrics_filtered.tsv'
UNFILTERED_FILENAME = 'best_model_metrics_unfiltered.tsv'
OUTPUT_DIR = "../../outputs/analysis_codes/lp"
NAMES = ['head', 'relation', 'tail']


def plot_dataset_ablation_study(unfiltered=False):
    # test set
    df_test = pd.read_csv(os.path.join(KGC_DATA_DIR, 'test.txt'), sep='\t', names=NAMES)

    dataset_folders = sorted(glob(os.path.join(RESULTS_ROOT_DIR, '*')))
    dataset_folders += [CHATGPT_METRICS_FILEPATH]

    data_dict = {
        'dataset': [],
        'model': [],
        'relations': [],
        'raw_precision_score': [],
        'raw_recall_score': [],
        'raw_f1_score': [],
        'precision_score': [],
        'recall_score': [],
        'f1_score': [],
        'accuracy_score': [],
        'average_precision_score': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'tp': [],
        'roc_auc_score': [],
        'best_model_df': [],
    }

    dataset_count = 0
    for dataset_folder in dataset_folders:
        if 'pykeen' in dataset_folder:
            dataset_str = Path(dataset_folder).name
            print(f'Processing dataset: {dataset_str}')
            model_folders = sorted(glob(os.path.join(dataset_folder, 'hpo/*')))

            data_folder = dataset_folder.replace('/pykeen/', '/data/')
            df_train = pd.read_csv(os.path.join(data_folder, 'train.txt'), sep='\t', names=NAMES)
            df_relation = pd.read_csv(os.path.join(data_folder, 'relations.txt'), sep='\t')
            relation_dict = dict(zip(
                df_relation['foodatlas_id'].tolist(), df_relation['name'].tolist()
            ))
            relation_dict_reverse = dict(zip(
                df_relation['name'].tolist(), df_relation['foodatlas_id'].tolist()
            ))
            relations = Counter(df_train['relation'].tolist())
            relations = {relation_dict[k]: v for k, v in relations.items()}
            print('    Relations: ', relations)
            print('    Train size: ', df_train.shape[0])

            train_entities = set(df_train['head'].tolist() + df_train['tail'].tolist())
            print('    Train entities size: ', len(train_entities))

            df_train_contains = df_train[df_train['relation'] == relation_dict_reverse['contains']]
            contains_food_entities = set(df_train_contains['head'].tolist())
            contains_chemical_entities = set(df_train_contains['tail'].tolist())
            print('    contains_food_entities: ', len(contains_food_entities))
            print('    contains_chemical_entities: ', len(contains_chemical_entities))

            def _f(row):
                if row['head'] not in train_entities or row['tail'] not in train_entities:
                    return False
                else:
                    return True

            df_test_unfiltered = df_test[df_test.apply(lambda row: _f(row), axis=1)]
            test_set_coverage = (df_test_unfiltered.shape[0] / df_test.shape[0]) * 100
            print('    ', test_set_coverage)

            best_model = ''
            best_model_f1 = 0
            best_model_df = None
            for model_folder in model_folders:
                model_str = Path(model_folder).name
                df_filtered = pd.read_csv(
                    os.path.join(model_folder, FILTERED_FILENAME), sep='\t')
                df_unfiltered = pd.read_csv(
                    os.path.join(model_folder, UNFILTERED_FILENAME), sep='\t')
                if unfiltered:
                    df_metrics = df_unfiltered
                else:
                    df_metrics = df_filtered
                f1_score = np.mean(df_metrics['f1_score'].tolist())
                if f1_score > best_model_f1:
                    best_model_f1 = f1_score
                    best_model = model_str
                    best_model_df = df_metrics
        elif 'chatgpt' in dataset_folder:
            best_model_df = pd.read_csv(os.path.join(dataset_folder, 'metrics.tsv'), sep='\t')
            dataset_str = 'None'
            best_model = 'ChatGPT'
            relations = {}
        else:
            raise RuntimeError()

        precision_score = best_model_df['precision_score'].tolist()
        recall_score = best_model_df['recall_score'].tolist()
        f1_score = best_model_df['f1_score'].tolist()
        accuracy_score = best_model_df['accuracy_score'].tolist()
        average_precision_score = best_model_df['average_precision_score'].tolist()
        tn = best_model_df['tn'].tolist()
        fp = best_model_df['fp'].tolist()
        fn = best_model_df['fn'].tolist()
        tp = best_model_df['tp'].tolist()
        roc_auc_score = best_model_df['roc_auc_score'].tolist()

        data_dict['dataset'].append(dataset_str)
        data_dict['model'].append(best_model)
        data_dict['relations'].append(relations)

        data_dict['raw_precision_score'].append(precision_score)
        data_dict['raw_recall_score'].append(recall_score)
        data_dict['raw_f1_score'].append(f1_score)

        data_dict['precision_score'].append(
            f'{np.mean(precision_score)*100:.2f}±{np.std(precision_score)*100:.2f}')
        data_dict['recall_score'].append(
            f'{np.mean(recall_score)*100:.2f}±{np.std(recall_score)*100:.2f}')
        data_dict['f1_score'].append(
            f'{np.mean(f1_score)*100:.2f}±{np.std(f1_score)*100:.2f}')
        data_dict['accuracy_score'].append(
            f'{np.mean(accuracy_score)*100:.2f}±{np.std(accuracy_score)*100:.2f}')
        data_dict['average_precision_score'].append(
            f'{np.mean(average_precision_score)*100:.2f}±{np.std(average_precision_score)*100:.2f}')
        data_dict['tn'].append(
            f'{np.mean(tn)*100:.2f}±{np.std(tn)*100:.2f}')
        data_dict['fp'].append(
            f'{np.mean(fp)*100:.2f}±{np.std(fp)*100:.2f}')
        data_dict['fn'].append(
            f'{np.mean(fn)*100:.2f}±{np.std(fn)*100:.2f}')
        data_dict['tp'].append(
            f'{np.mean(tp)*100:.2f}±{np.std(tp)*100:.2f}')
        data_dict['roc_auc_score'].append(
            f'{np.mean(roc_auc_score)*100:.2f}±{np.std(roc_auc_score)*100:.2f}')
        data_dict['best_model_df'].append(
            best_model_df)

        dataset_count += 1

    filter = 'unfiltered' if unfiltered else 'filtered'
    df = pd.DataFrame(data_dict).drop('best_model_df', axis=1)
    df.to_csv(os.path.join(OUTPUT_DIR, f'dataset_ablation_df_{filter}.txt'), sep='\t')
    print(df)

    fig = go.Figure()
    for metric in ['precision_score', 'recall_score', 'f1_score']:
        print(f'Metric: {metric}')

        x, y = [], []
        for idx in range(dataset_count):
            dataset_str = data_dict['dataset'][idx]
            best_model = data_dict['model'][idx]
            best_model_df = data_dict['best_model_df'][idx]
            x.extend(best_model_df[metric].tolist())
            y.extend([dataset_str] * best_model_df.shape[0])

        fig.add_trace(go.Box(x=y, y=x, name=metric))

    fig.update_layout(boxmode='group')
    # fig.update_traces(orientation='h')
    fig.update_layout(
        font_family="Arial",
        height=450, width=1000,
    )
    fig.update_xaxes(
        categoryorder='array',
        categoryarray=[
            'annotations',
            'annotations_extdb',
            'annotations_mesh_ncbi',
            'annotations_predictions',
            'annotations_extdb_mesh_ncbi',
            'annotations_predictions_extdb',
            'annotations_predictions_mesh_ncbi',
            'annotations_predictions_extdb_mesh_ncbi',
            'annotations_predictions-above60_extdb_mesh_ncbi',
            'annotations_predictions-above70_extdb_mesh_ncbi',
            'annotations_predictions-above80_extdb_mesh_ncbi',
            'annotations_predictions-above90_extdb_mesh_ncbi',
        ]
    )
    fig.write_image(os.path.join(OUTPUT_DIR, f'dataset_ablation_box_{filter}.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'dataset_ablation_box_{filter}.svg'))


def model_rank_heatmap():
    dataset_folders = sorted(glob(os.path.join(RESULTS_ROOT_DIR, '*')))

    data_f1 = {}
    data_precision = {}
    data_recall = {}
    index = []
    for dataset_folder in dataset_folders:
        dataset_str = Path(dataset_folder).name
        print(f'Processing dataset: {dataset_str}')
        index.append(dataset_str)

        model_folders = sorted(glob(os.path.join(dataset_folder, 'hpo/*')))
        for model_folder in model_folders:
            model_str = Path(model_folder).name

            df_metrics = pd.read_csv(
                os.path.join(model_folder, UNFILTERED_FILENAME), sep='\t')
            f1_score = np.mean(df_metrics['f1_score'].tolist()) * 100
            precision_score = np.mean(df_metrics['precision_score'].tolist()) * 100
            recall_score = np.mean(df_metrics['recall_score'].tolist()) * 100

            if model_str not in data_f1:
                data_f1[model_str] = [f1_score]
            else:
                data_f1[model_str].append(f1_score)

            if model_str not in data_precision:
                data_precision[model_str] = [precision_score]
            else:
                data_precision[model_str].append(precision_score)

            if model_str not in data_recall:
                data_recall[model_str] = [recall_score]
            else:
                data_recall[model_str].append(recall_score)

    data_dict = {
        'f1': data_f1,
        'precision': data_precision,
        'recall': data_recall,
    }

    for metric, data in data_dict.items():
        df = pd.DataFrame(data, index=index).transpose()
        df = df.rank(ascending=False)

        column_mapping = {
            'annotations': 'FA_A',
            'annotations_extdb': 'FA_AE',
            'annotations_extdb_mesh_ncbi': 'FA_AER',
            'annotations_mesh_ncbi': 'FA_AR',
            'annotations_predictions': 'FA_AP',
            'annotations_predictions-above60_extdb_mesh_ncbi': 'FA_AERP60',
            'annotations_predictions-above70_extdb_mesh_ncbi': 'FA_AERP70',
            'annotations_predictions-above80_extdb_mesh_ncbi': 'FA_AERP80',
            'annotations_predictions-above90_extdb_mesh_ncbi': 'FA_AERP90',
            'annotations_predictions_extdb': 'FA_AEP',
            'annotations_predictions_extdb_mesh_ncbi': 'FA_AERP50',
            'annotations_predictions_mesh_ncbi': 'FA_ARP',
        }

        df.rename(column_mapping, axis=1, inplace=True)
        df = df[[
            'FA_A', 'FA_AE', 'FA_AR', 'FA_AP', 'FA_AER', 'FA_AEP', 'FA_ARP',
            'FA_AERP50', 'FA_AERP60', 'FA_AERP70', 'FA_AERP80', 'FA_AERP90',
        ]]

        plt.figure(figsize=(14, 2.75))
        if metric == 'f1':
            cmap = 'Greens_r'
        elif metric == 'precision':
            cmap = 'Blues_r'
        elif metric == 'recall':
            cmap = 'Reds_r'
        else:
            raise ValueError()
        heatmap = sns.heatmap(df, annot=True, fmt='.0f', cmap=cmap)
        fig = heatmap.get_figure()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'model_{metric}_heatmap.png'))
        fig.savefig(os.path.join(OUTPUT_DIR, f'model_{metric}_heatmap.svg'))


def plot_rank_based_metrics():
    df_metrics = pd.read_csv(
        os.path.join(
            RESULTS_ROOT_DIR,
            'annotations_predictions-above80_extdb_mesh_ncbi',
            'hpo', 'RotatE', 'rank_based_metrics.tsv',
        ),
        sep='\t',
    )

    head_mr = df_metrics['head.realistic.arithmetic_mean_rank'].tolist()
    tail_mr = df_metrics['tail.realistic.arithmetic_mean_rank'].tolist()
    both_mr = df_metrics['both.realistic.arithmetic_mean_rank'].tolist()
    print('head_mr: ', head_mr)
    print('tail_mr: ', tail_mr)
    print('both_mr: ', both_mr)
    print(f'head_mr: {np.mean(head_mr):.3f} ± {np.std(head_mr):.3f}')
    print(f'tail_mr: {np.mean(tail_mr):.3f} ± {np.std(tail_mr):.3f}')
    print(f'both_mr: {np.mean(both_mr):.3f} ± {np.std(both_mr):.3f}')

    head_mrr = df_metrics['head.realistic.inverse_harmonic_mean_rank'].tolist()
    tail_mrr = df_metrics['tail.realistic.inverse_harmonic_mean_rank'].tolist()
    both_mrr = df_metrics['both.realistic.inverse_harmonic_mean_rank'].tolist()
    print('head_mrr: ', head_mrr)
    print('tail_mrr: ', tail_mrr)
    print('both_mrr: ', both_mrr)
    print(f'head_mrr: {np.mean(head_mrr):.3f} ± {np.std(head_mrr):.3f}')
    print(f'tail_mrr: {np.mean(tail_mrr):.3f} ± {np.std(tail_mrr):.3f}')
    print(f'both_mrr: {np.mean(both_mrr):.3f} ± {np.std(both_mrr):.3f}')

    head_hits_at_1 = df_metrics['head.realistic.hits_at_1'].tolist()
    tail_hits_at_1 = df_metrics['tail.realistic.hits_at_1'].tolist()
    both_hits_at_1 = df_metrics['both.realistic.hits_at_1'].tolist()
    print('head_hits_at_1: ', head_hits_at_1)
    print('tail_hits_at_1: ', tail_hits_at_1)
    print('both_hits_at_1: ', both_hits_at_1)
    print(f'head_hits_at_1: {np.mean(head_hits_at_1):.3f} ± {np.std(head_hits_at_1):.3f}')
    print(f'tail_hits_at_1: {np.mean(tail_hits_at_1):.3f} ± {np.std(tail_hits_at_1):.3f}')
    print(f'both_hits_at_1: {np.mean(both_hits_at_1):.3f} ± {np.std(both_hits_at_1):.3f}')

    head_hits_at_3 = df_metrics['head.realistic.hits_at_3'].tolist()
    tail_hits_at_3 = df_metrics['tail.realistic.hits_at_3'].tolist()
    both_hits_at_3 = df_metrics['both.realistic.hits_at_3'].tolist()
    print('head_hits_at_3: ', head_hits_at_3)
    print('tail_hits_at_3: ', tail_hits_at_3)
    print('both_hits_at_3: ', both_hits_at_3)
    print(f'head_hits_at_3: {np.mean(head_hits_at_3):.3f} ± {np.std(head_hits_at_3):.3f}')
    print(f'tail_hits_at_3: {np.mean(tail_hits_at_3):.3f} ± {np.std(tail_hits_at_3):.3f}')
    print(f'both_hits_at_3: {np.mean(both_hits_at_3):.3f} ± {np.std(both_hits_at_3):.3f}')

    head_hits_at_10 = df_metrics['head.realistic.hits_at_10'].tolist()
    tail_hits_at_10 = df_metrics['tail.realistic.hits_at_10'].tolist()
    both_hits_at_10 = df_metrics['both.realistic.hits_at_10'].tolist()
    print('head_hits_at_10: ', head_hits_at_10)
    print('tail_hits_at_10: ', tail_hits_at_10)
    print('both_hits_at_10: ', both_hits_at_10)
    print(f'head_hits_at_10: {np.mean(head_hits_at_10):.3f} ± {np.std(head_hits_at_10):.3f}')
    print(f'tail_hits_at_10: {np.mean(tail_hits_at_10):.3f} ± {np.std(tail_hits_at_10):.3f}')
    print(f'both_hits_at_10: {np.mean(both_hits_at_10):.3f} ± {np.std(both_hits_at_10):.3f}')
    sys.exit()

    fig = go.Figure()
    fig.add_trace(go.Box(x=['head_mr']*len(head_mr), y=head_mr))
    fig.add_trace(go.Box(x=['tail_mr']*len(tail_mr), y=tail_mr))
    fig.add_trace(go.Box(x=['both_mr']*len(both_mr), y=both_mr))
    fig.update_layout(height=500, width=350)
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_mr.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_mr.svg'))

    fig = go.Figure()
    fig.add_trace(go.Box(x=['head_mrr']*len(head_mrr), y=head_mrr))
    fig.add_trace(go.Box(x=['tail_mrr']*len(tail_mrr), y=tail_mrr))
    fig.add_trace(go.Box(x=['both_mrr']*len(both_mrr), y=both_mrr))
    fig.update_layout(yaxis_range=[0.15, 0.65], height=500, width=350)
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_mrr.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_mrr.svg'))

    fig = go.Figure()
    fig.add_trace(go.Box(x=['head_hits_at_1']*len(head_hits_at_1), y=head_hits_at_1))
    fig.add_trace(go.Box(x=['tail_hits_at_1']*len(tail_hits_at_1), y=tail_hits_at_1))
    fig.add_trace(go.Box(x=['both_hits_at_1']*len(both_hits_at_1), y=both_hits_at_1))
    fig.update_layout(yaxis_range=[0.15, 0.65], height=500, width=350)
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits1.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits1.svg'))

    fig = go.Figure()
    fig.add_trace(go.Box(x=['head_hits_at_3']*len(head_hits_at_3), y=head_hits_at_3))
    fig.add_trace(go.Box(x=['tail_hits_at_3']*len(tail_hits_at_3), y=tail_hits_at_3))
    fig.add_trace(go.Box(x=['both_hits_at_3']*len(both_hits_at_3), y=both_hits_at_3))
    fig.update_layout(yaxis_range=[0.15, 0.65], height=500, width=350)
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits3.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits3.svg'))

    fig = go.Figure()
    fig.add_trace(go.Box(x=['head_hits_at_10']*len(head_hits_at_10), y=head_hits_at_10))
    fig.add_trace(go.Box(x=['tail_hits_at_10']*len(tail_hits_at_10), y=tail_hits_at_10))
    fig.add_trace(go.Box(x=['both_hits_at_10']*len(both_hits_at_10), y=both_hits_at_10))
    fig.update_layout(yaxis_range=[0.15, 0.65], height=500, width=350)
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits10.png'))
    fig.write_image(os.path.join(OUTPUT_DIR, f'rank_based_metrics_hits10.svg'))


def calculate_pval():
    df = pd.read_csv(
        '../../outputs/analysis_codes/lp/dataset_ablation_df_unfiltered.txt',
        sep='\t',
    )
    df['raw_precision_score'] = df['raw_precision_score'].apply(literal_eval)
    df['raw_recall_score'] = df['raw_recall_score'].apply(literal_eval)
    df['raw_f1_score'] = df['raw_f1_score'].apply(literal_eval)
    df.set_index('dataset', inplace=True)

    # A vs A,E
    print('A vs A,E')
    a_f1 = df.at['annotations', 'raw_f1_score']
    b_f1 = df.at['annotations_extdb', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,E: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A,R,P vs A,E,R,P50
    print('A,R,P vs A,E,R,P50')
    a_f1 = df.at['annotations_predictions_mesh_ncbi', 'raw_f1_score']
    b_f1 = df.at['annotations_predictions_extdb_mesh_ncbi', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A,R,P: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,E,R,P50: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A vs A,R
    print('A vs A,R')
    a_f1 = df.at['annotations', 'raw_f1_score']
    b_f1 = df.at['annotations_mesh_ncbi', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,R: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A,E,P vs A,E,R,P50
    print('A,E,P vs A,E,R,P50')
    a_f1 = df.at['annotations_predictions_extdb', 'raw_f1_score']
    b_f1 = df.at['annotations_predictions_extdb_mesh_ncbi', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A,E,P: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,E,R,P50: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A vs A,P
    print('A vs A,P')
    a_f1 = df.at['annotations', 'raw_f1_score']
    b_f1 = df.at['annotations_predictions', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,P: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A,E,R vs A,E,R,P50
    print('A,E,R vs A,E,R,P50')
    a_f1 = df.at['annotations_extdb_mesh_ncbi', 'raw_f1_score']
    b_f1 = df.at['annotations_predictions_extdb_mesh_ncbi', 'raw_f1_score']

    _, pval = ttest_ind(a_f1, b_f1)
    print(f'A,E,R: {np.mean(a_f1)*100:.1f} ± {np.std(a_f1)*100:.1f}')
    print(f'A,E,R,P50: {np.mean(b_f1)*100:.1f} ± {np.std(b_f1)*100:.1f}')
    print(f'pval: {pval}')
    print()

    # A,E,R,P80
    print('A,E,R,P80')
    precision = df.at['annotations_predictions-above80_extdb_mesh_ncbi', 'raw_precision_score']
    recall = df.at['annotations_predictions-above80_extdb_mesh_ncbi', 'raw_recall_score']
    f1 = df.at['annotations_predictions-above80_extdb_mesh_ncbi', 'raw_f1_score']
    print(f'precision: {np.mean(precision)*100:.1f} ± {np.std(precision)*100:.1f}')
    print(f'recall: {np.mean(recall)*100:.1f} ± {np.std(recall)*100:.1f}')
    print(f'f1: {np.mean(f1)*100:.1f} ± {np.std(f1)*100:.1f}')

    # A,R
    print('A,R')
    precision = df.at['annotations_mesh_ncbi', 'raw_precision_score']
    recall = df.at['annotations_mesh_ncbi', 'raw_recall_score']
    f1 = df.at['annotations_mesh_ncbi', 'raw_f1_score']
    print(f'precision: {np.mean(precision)*100:.1f} ± {np.std(precision)*100:.1f}')
    print(f'recall: {np.mean(recall)*100:.1f} ± {np.std(recall)*100:.1f}')
    print(f'f1: {np.mean(f1)*100:.1f} ± {np.std(f1)*100:.1f}')

    # ChatGPT
    print('ChatGPT')
    precision = df.at['None', 'raw_precision_score']
    recall = df.at['None', 'raw_recall_score']
    f1 = df.at['None', 'raw_f1_score']
    print(f'precision: {np.mean(precision)*100:.1f} ± {np.std(precision)*100:.1f}')
    print(f'recall: {np.mean(recall)*100:.1f} ± {np.std(recall)*100:.1f}')
    print(f'f1: {np.mean(f1)*100:.1f} ± {np.std(f1)*100:.1f}')


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # plot_dataset_ablation_study(unfiltered=True)
    # plot_dataset_ablation_study(unfiltered=False)
    # model_rank_heatmap()
    # plot_rank_based_metrics()
    calculate_pval()


if __name__ == '__main__':
    main()
