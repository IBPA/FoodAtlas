import argparse
from ast import literal_eval
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tqdm import tqdm

QUERY_TEMPLATE = '{food} AND ((chemical) OR (compound) OR (molecule) OR (nutrient) OR (metabolite))'


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Single query (cocoa), multiple queries (cocoa,banana), or '
             '.txt file containing one query for each line.',
    )

    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='pmc, pubmed, or both',
    )

    parser.add_argument(
        '--query_uid_results_filepath',
        type=str,
        # required=True,
        default='../../outputs/pmc_pubmed/query_uid_results.tsv',
        help='Output directory',
    )

    return parser.parse_args()


def parse_query(query):
    if query.endswith('.txt'):
        df = pd.read_csv(query, sep='\t', keep_default_na=False, names=['query'])
        queries = df['query'].tolist()
    elif ',' in query:
        queries = query.split(',')
    else:
        queries = [query]

    print(f'Got {len(queries)} queries...')

    return queries


def parse_db(db):
    if db == 'pmc':
        return ['pmc']
    elif db == 'pubmed':
        return ['pubmed']
    elif db == 'both':
        return ['pmc', 'pubmed']
    else:
        raise ValueError()


def get_pmcid_pmid_mapping(filepath='../../data/NCBI/PMC-ids.csv'):
    print('Fetching PMCID-PMID mapping...')
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    pmcid_pmid_dict = dict(zip(df['PMCID'], df['PMID']))
    pmid_pmcid_dict = dict(zip(df['PMID'], df['PMCID']))
    return pmcid_pmid_dict, pmid_pmcid_dict


def load_data(filepath):
    if not Path(filepath).is_file():
        return {}

    df = pd.read_csv(filepath, sep='\t', dtype=str, keep_default_na=False)
    df['key'] = tuple(zip(df['pmid'], df['pmcid']))
    df['queries'] = df['queries'].apply(literal_eval)

    return dict(zip(df['key'], df['queries']))


def save_data(data, filepath):
    df = pd.DataFrame({'key': data.keys(), 'queries': data.values()})
    df['pmid'] = df['key'].apply(lambda x: x[0])
    df['pmcid'] = df['key'].apply(lambda x: x[1])

    df = df[['pmid', 'pmcid', 'queries']]
    df.to_csv(filepath, sep='\t', index=False)


def main():
    args = parse_argument()

    Path(args.query_uid_results_filepath).parent.mkdir(exist_ok=True, parents=True)

    pmcid_pmid_dict, pmid_pmcid_dict = get_pmcid_pmid_mapping()
    data = load_data(args.query_uid_results_filepath)  # (pmid,pmcid): [queries]

    queries = parse_query(args.query)
    dbs = parse_db(args.db)

    for db in dbs:
        print(f'Searching {db}...')

        pbar = tqdm(queries)
        for q in pbar:
            pbar.set_description(f'Processing {q}')

            try:
                esearch_args = ('esearch', '-db', db, '-query', QUERY_TEMPLATE.format(food=q))
                esearch = subprocess.Popen(esearch_args, stdout=subprocess.PIPE)
                efetch_args = ('efetch', '-format', 'uid')
                result = subprocess.check_output(efetch_args, stdin=esearch.stdout, text=True)
                esearch.wait()
            except Exception:
                continue

            if db == 'pmc':
                ids = [f'PMC{id}' for id in result.rstrip().split('\n')]
            else:
                ids = [id for id in result.rstrip().split('\n')]

            # key should be a tuple (pmid,pmcid)
            for id in ids:
                if db == 'pmc':
                    if id in pmcid_pmid_dict:
                        key = (pmcid_pmid_dict[id], id)
                    else:
                        key = ('', id)

                if db == 'pubmed':
                    if id in pmid_pmcid_dict:
                        key = (id, pmid_pmcid_dict[id])
                    else:
                        key = (id, '')

                if key in data and q not in data[key]:
                    data[key].append(q)
                else:
                    data[key] = [q]

    save_data(data, args.query_uid_results_filepath)


if __name__ == '__main__':
    main()
