import pandas as pd

files = [
    '../../outputs/data_processing/predicted.csv',
    '../../outputs/data_processing/predictions_ph_pairs_20230118_144448.tsv',
    '../../outputs/data_processing/predictions_ph_pairs_additional.txt.tsv',
    '../../outputs/data_processing/ph_pairs_20230111_224704_predicted.tsv',
    '../../outputs/data_processing/ph_pairs_20230112_114749_predicted.tsv',
    '../../outputs/data_processing/ph_pairs_20230119_214855_predicted.tsv',
]

data = []
for f in files:
    if f.endswith('.csv'):
        data.append(pd.read_csv(f))
    else:
        data.append(pd.read_csv(f, sep='\t'))

df = pd.concat(data)
print(df)
df = df[df['head'].apply(lambda x: "['3641']" in x or "['4682']" in x)]
print(df)
df = df[df['prob_mean'] < 0.5]
df.to_csv('../../outputs/analysis_codes/garlic_cocoa_negatives.tsv', sep='\t', index=False)
