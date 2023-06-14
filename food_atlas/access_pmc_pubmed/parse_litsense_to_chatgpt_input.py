from glob import glob
import json
import os
from pathlib import Path
import pickle
import sys

import pandas as pd
from tqdm import tqdm

query_dir = '../../data/FoodAtlas/litsense_query/queries_output/'
json_files = sorted(glob(os.path.join(query_dir, '*')))
json_files = [x for x in json_files if not x.endswith('summary.txt')]

data = {
    'query': [],
    'pmid': [],
    'pmcid': [],
    'section': [],
    'text': [],
    'annotations': [],
    'score': [],
}

for filepath in tqdm(json_files):
    with open(filepath) as _f:
        response = _f.readlines()

    query_str = response[0].strip()
    response = response[1:]

    assert len(response) == 1
    response = json.loads(response[0])

    for r in response:
        data['query'].append(query_str)
        data['pmid'].append(r['pmid'])
        data['pmcid'].append(r['pmcid'])
        data['section'].append(r['section'])
        data['text'].append(r['text'])
        data['annotations'].append(r['annotations'])
        data['score'].append(r['score'])

df = pd.DataFrame(data)
df.drop_duplicates(['pmid', 'pmcid', 'text'], inplace=True)

with open('../../data/FoodAtlas/premises_from_litsense.pkl', 'wb') as _f:
    pickle.dump(df, _f)
