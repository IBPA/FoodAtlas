import os
import sys

sys.path.append('../data_processing/')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

fa_kg = KnowledgeGraph('../../outputs/backend_data/v0.1')

df_evidence = fa_kg.get_evidence()

pmids = set(df_evidence['pmid'].tolist())
print(len(pmids))

df_temp = df_evidence[df_evidence['premise'].apply(
    lambda x: "is a fruit known for its high content of vitamin C" in x
)]

df_temp = df_temp[df_temp['relation'] == 'r1']

