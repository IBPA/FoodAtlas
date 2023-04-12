import argparse
from ast import literal_eval
import json
import os
from pathlib import Path
import requests
import sys

sys.path.append('../../data_processing/')
sys.path.append('../')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.utils import load_pkl  # noqa: E402
from utils.utils import generate_report  # noqa: E402
from merge_ncbi_taxonomy import read_dmp_files  # noqa: E402
from common_utils.chemical_db_ids import (  # noqa: E402
    _get_name_from_json, get_mesh_name_using_mesh_id,
    read_mesh_data
)


URL = 'https://api.openai.com/v1/completions'
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
NAMES = ['head', 'relation', 'tail', 'label']
QUESTION = 'Does {}? Answer Yes or No.'
N = 5
ANSWERS_FILENAME = 'answers.tsv'
METRICS_FILENAME = 'metrics.tsv'

NCBI_NAMES_FILEPATH = "../../../data/NCBI_Taxonomy/names.dmp"
# NCBI_NAME_CLASS_TO_USE = ["genbank common name", "scientific name", "common name"]
NCBI_NAME_CLASS_TO_USE = ["scientific name"]
CID_JSON_LOOKUP_PKL_FILEPATH = "../../../data/FoodAtlas/cid_json_lookup.pkl"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--kg_dir",
        type=str,
        # required=True,
        default='../../../outputs/kg/annotations_predictions_extdb_mesh_ncbi',
        help="KG directory",
    )

    parser.add_argument(
        "--test_filepath",
        type=str,
        # required=True,
        default='../../../outputs/kgc/data/test_corrupted.txt',
        help="test.txt filepath",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default='../../../outputs/kgc/chatgpt',
        help="Output directory",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    fa_kg = KnowledgeGraph(args.kg_dir)
    df_test = pd.read_csv(args.test_filepath, sep='\t', names=NAMES)

    # load entity name lookup files
    print('Loading NCBI names...')
    df_names = read_dmp_files(NCBI_NAMES_FILEPATH, filetype="names")
    df_names = df_names[df_names["name_class"].apply(lambda x: x in NCBI_NAME_CLASS_TO_USE)]
    df_names = df_names.groupby("tax_id")["name_txt"].apply(list).reset_index()
    names_lookup = dict(zip(
        df_names['tax_id'].tolist(), df_names['name_txt'].tolist()
    ))

    print('Loading cid_json_lookup...')
    cid_json_lookup = load_pkl(CID_JSON_LOOKUP_PKL_FILEPATH)

    print('Loading MeSH data...')
    mesh_data_dict = read_mesh_data()

    rows = []
    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        head = fa_kg.get_entity_by_id(row['head'])
        relation = fa_kg.get_relation_by_id(row['relation'])
        tail = fa_kg.get_entity_by_id(row['tail'])

        assert head['other_db_ids']['foodatlas_part_id'] == 'p0'
        assert relation['name'] == 'contains'

        ncbi_id = head['other_db_ids']['NCBI_taxonomy'][0]
        if ncbi_id in names_lookup:
            head_str = names_lookup[ncbi_id][0]
        else:
            head_str = head['name']
        if 'PubChem' in tail['other_db_ids']:
            tail_str = _get_name_from_json(cid_json_lookup[tail['other_db_ids']['PubChem'][0]])
        elif 'MESH' in tail['other_db_ids']:
            mesh_id = tail['other_db_ids']['MESH'][0]
            tail_str = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
        else:
            raise RuntimeError()

        data = {
            'model': 'text-davinci-003',
            'prompt': QUESTION.format(f'{head_str} contain {tail_str}'),
            'n': N,
        }

        print(QUESTION.format(f'{head_str} contain {tail_str}'))
        continue

        response = requests.post(URL, headers=HEADERS, json=data).json()
        # assert len(response['choices']) == N
        # for i, choice in enumerate(response['choices']):
        #     row[f'answer_{i}'] = choice['text'].strip()

        row['response'] = response
        rows.append(row)

    sys.exit()

    df_test_with_answers = pd.DataFrame(rows)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df_test_with_answers.to_csv(
        os.path.join(args.output_dir, ANSWERS_FILENAME),
        sep='\t', index=False,
    )

    df_test_with_answers = pd.read_csv(
        os.path.join(args.output_dir, ANSWERS_FILENAME),
        sep='\t',
    )

    rows = []
    for _, row in tqdm(df_test_with_answers.iterrows(), total=df_test_with_answers.shape[0]):
        response = literal_eval(row['response'])
        assert len(response['choices']) == N
        for i, choice in enumerate(response['choices']):
            answer = choice['text'].strip()
            if 'yes' in answer.lower():
                row[f'answer_{i}'] = 1
            elif 'no' in answer.lower():
                row[f'answer_{i}'] = 0
            else:
                raise RuntimeError()
        rows.append(row)

    df_test_with_answers = pd.DataFrame(rows)
    print(df_test_with_answers)

    metrics = []
    y_true = df_test_with_answers['label'].tolist()
    for i in range(N):
        y_pred = df_test_with_answers[f'answer_{i}'].tolist()
        metrics.append(generate_report(y_true, y_pred))

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(
        os.path.join(args.output_dir, METRICS_FILENAME),
        sep='\t', index=False
    )


if __name__ == '__main__':
    main()
