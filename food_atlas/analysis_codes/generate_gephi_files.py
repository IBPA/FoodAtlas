from pathlib import Path
import sys

sys.path.append('../data_processing/')

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

fa_kg = KnowledgeGraph(kg_dir='../../outputs/kg/annotations_predictions_extdb_mesh_ncbi/')

df_entities = fa_kg.get_all_entities()
df_entities['Id'] = df_entities['foodatlas_id']
df_entities['Label'] = df_entities['name']
df_entities['Type'] = df_entities['type'].apply(lambda x: x.split(':')[0])
df_entities = df_entities[['Id', 'Label', 'Type']]
print(df_entities)

df_relations = fa_kg.get_all_relations()
relation_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

df_kg = fa_kg.get_kg()
df_kg['Source'] = df_kg['head']
df_kg['Target'] = df_kg['tail']
df_kg['Label'] = df_kg['relation'].apply(lambda x: relation_dict[x])
df_kg['Type'] = 'Undirected'
df_kg['Weight'] = 1
df_kg = df_kg[['Source', 'Target', 'Label', 'Type', 'Weight']]
print(df_kg)

Path('../../outputs/analysis_codes/gephi').mkdir(exist_ok=True, parents=True)
df_entities.to_csv('../../outputs/analysis_codes/gephi/nodes.csv', sep=';', index=False)
df_kg.to_csv('../../outputs/analysis_codes/gephi/edges.csv', sep=';', index=False)
