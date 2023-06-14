import os
import sys
from collections import Counter

sys.path.append('../data_processing/')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

fa_kg = KnowledgeGraph('../../outputs/backend_data/v0.1')

df_evidence = fa_kg.get_evidence()
print(df_evidence)

df = df_evidence.groupby(['head', 'relation', 'tail'])['quality'].apply(list).reset_index()

def _choose_quality(x):
    if 'high' in x:
        return 'high'
    if 'medium' in x:
        return 'medium'
    if 'low' in x:
        return 'low'


df['quality'] = df['quality'].apply(lambda x: _choose_quality(x))
print(df)

print(Counter(df['quality'].tolist()))


print(set(df_evidence['source'].tolist()))
df = df_evidence[df_evidence['source'].apply(lambda x: x.startswith('FoodAtlas'))]
print(df)

df = df.groupby(['head', 'relation', 'tail'])['quality'].apply(list).reset_index()
df['quality'] = df['quality'].apply(lambda x: _choose_quality(x))
print(df)

print(Counter(df['quality'].tolist()))
