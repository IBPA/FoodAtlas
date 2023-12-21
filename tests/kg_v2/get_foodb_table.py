import pandas as pd


if __name__ == '__main__':
    data = pd.read_json("tests/kg_v2/data/FooDB/Food.json", lines=True)

    data = data[[
        'id', 'public_id', 'name', 'name_scientific', 'itis_id', 'wikipedia_id',
        'food_group', 'food_subgroup', 'food_type', 'category', 'ncbi_taxonomy_id',
    ]]
    data.to_csv('tests/kg_v2/outputs/foodb.csv')
