import pandas as pd


if __name__ == '__main__':
    chemicals = pd.read_csv(
        "data/FooDB/foodb_2020_04_07_csv/Compound.csv", low_memory=False)
    chemicals = chemicals.rename(
        columns={'moldb_smiles': 'inchikey'})
    chemicals = chemicals.query("inchikey.notnull()")
    inchikeys = chemicals['inchikey'].unique().tolist()

    with open("outputs/benchmark/inchikeys_foodb.txt", 'w') as f:
        f.write(' '.join(inchikeys))
