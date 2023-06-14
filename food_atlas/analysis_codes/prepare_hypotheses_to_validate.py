import sys

sys.path.append('../data_processing')

import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

pandarallel.initialize(progress_bar=True)

df = pd.read_csv(
    '../../outputs/kgc/pykeen/annotations_predictions-' +
    'above80_extdb_mesh_ncbi/hpo/RotatE/hypotheses_20230323_010058.tsv',
    sep='\t')
print(df)

fa_kg = KnowledgeGraph('../../outputs/kg/annotations_predictions_extdb_mesh_ncbi')


#
def _f(row):
    result = True
    prob_names = [x for x in row.index.values.tolist() if x.startswith('pred_run_')]
    for x in prob_names:
        result &= row[x]
    return result


df['pred'] = df.parallel_apply(lambda row: _f(row), axis=1)
df_to_validate = df[df['pred']]
df_to_validate = df_to_validate[['head', 'relation', 'tail', 'prob_mean', 'prob_std']]
print(df_to_validate)

df_to_validate_lower = df.tail(180)[['head', 'relation', 'tail', 'prob_mean', 'prob_std']]
print(df_to_validate_lower)


df_previous = pd.read_csv('../../outputs/hypotheses/to_validate_random_300.tsv', sep='\t')


random_data = []
prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(prob_bins)-1):
    lower, upper = prob_bins[i], prob_bins[i+1]
    if lower != 0.9:
        df_temp = df[df['prob_mean'].apply(lambda x: x >= lower and x < upper)]
    else:
        df_temp = df[df['prob_mean'].apply(lambda x: x >= lower)]

    df_sampled = df_temp.sample(10)

    #
    while True:
        df_concat = pd.concat([df_sampled, df_previous])
        temp = df_concat.duplicated().any()
        if not temp:
            break

    random_data.append(df_sampled)
    print(f'{lower}-{upper}: {df_temp.shape[0]}')

df_random = pd.concat(random_data)[['head', 'relation', 'tail', 'prob_mean', 'prob_std']]
df_random.sort_values('prob_mean', ascending=False, inplace=True)


def _get_entity(row):
    head_info = fa_kg.get_entity_by_id(row['head']).to_dict()
    tail_info = fa_kg.get_entity_by_id(row['tail']).to_dict()
    head_info.pop('synonyms', None)
    tail_info.pop('synonyms', None)
    return [head_info, tail_info]


# df_to_validate[['head_info', 'tail_info']] = df_to_validate.parallel_apply(
#     lambda row: _get_entity(row), axis=1, result_type='expand')
# df_to_validate.to_csv(
#     '../../outputs/hypotheses/to_validate.tsv',
#     sep='\t', index=False,
# )

# df_to_validate_lower[['head_info', 'tail_info']] = df_to_validate_lower.parallel_apply(
#     lambda row: _get_entity(row), axis=1, result_type='expand')
# df_to_validate_lower.to_csv(
#     '../../outputs/hypotheses/to_validate_lowest_180.tsv',
#     sep='\t', index=False,
# )

df_random[['head_info', 'tail_info']] = df_random.parallel_apply(
    lambda row: _get_entity(row), axis=1, result_type='expand')
df_random.to_csv(
    '../../outputs/hypotheses/to_validate_random_100_additional.tsv',
    sep='\t', index=False,
)
