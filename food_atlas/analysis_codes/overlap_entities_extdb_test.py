import os
import sys
import pandas as pd

sys.path.append('../data_processing/')


from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

fa_kg = KnowledgeGraph(kg_dir='../../outputs/kg/annotations_predictions_extdb_mesh_ncbi/')

df_evidence = fa_kg.get_evidence()
print(df_evidence)

sources = set(df_evidence['source'].tolist())
print(sources)

external_dbs = ['Phenol-Explorer', 'Frida', 'FDC']
df_extdb = df_evidence[df_evidence['source'].apply(lambda x: x in external_dbs)]
print(df_extdb)

extdb_entities = set(df_extdb['head'].tolist() + df_extdb['tail'].tolist())

df_test = pd.read_csv('../../outputs/kgc/data/test.txt', names=['head', 'relation', 'tail'])
test_entities = set(df_test['head'].tolist() + df_test['tail'].tolist())

print(len(extdb_entities))
print(len(test_entities))

common_entities = extdb_entities.intersection(test_entities)
print(len(common_entities))

all_entities = extdb_entities.union(test_entities)
print(len(all_entities))
