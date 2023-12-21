import pandas as pd


def get_foundation_food():
    PATH_DATA_DIR \
        = "tests/kg_v2/data/FDC/FoodData_Central_foundation_food_csv_2023-10-26"

    data_food = pd.read_csv(f"{PATH_DATA_DIR}/food.csv")
    data_ff_ids = pd.read_csv(f"{PATH_DATA_DIR}/foundation_food.csv")['fdc_id'].tolist()
    data_ff = data_food[data_food['fdc_id'].isin(data_ff_ids)]
    data_ff_attr \
        = pd.read_csv(f"{PATH_DATA_DIR}/food_attribute.csv").set_index('fdc_id')

    def _extract_ff_food_attributes(row):
        result = {
            # 'common_names': [],
            'foodon_name': None,
            'foodon_url': None,
            'ncbi_taxon_url': None,
        }
        if row['fdc_id'] in data_ff_attr.index:
            attr = data_ff_attr.loc[row['fdc_id']]
            if type(attr) == pd.Series:
                attr = pd.DataFrame(attr).T
            attr = attr.set_index('name')

            for attr_name, col_name in zip(
                [
                    'FoodOn Ontology name for FDC item',
                    'FoodOn Ontology ID for FDC item',
                    'NCBI Taxon',
                ],
                [
                    'foodon_name',
                    'foodon_url',
                    'ncbi_taxon_url',
                ]
            ):
                result[col_name] = attr.loc[attr_name, 'value'] \
                    if attr_name in attr.index else None

    #     return pd.Series(result)

    # data_ff = pd.concat(
    #     [data_ff, data_ff.apply(_extract_ff_food_attributes, axis=1)],
    #     axis=1
    # )

    # return data_ff


def get_legacy_food():
    # # Load legacy food.
    data_sr = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food.csv",
    )
    data_sr_attr = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food_attribute.csv",
    ).set_index('fdc_id')

    def _extract_sr_food_attributes(row):
        result = {
            'common_names': [],
            'foodon_name': None,
            'foodon_url': None,
            'ncbi_taxon_url': None,
        }
        if row['fdc_id'] in data_sr_attr.index:
            attr = data_sr_attr.loc[row['fdc_id']]
            if type(attr) == pd.Series:
                attr = pd.DataFrame(attr).T
            attr = attr.set_index('name')

            result['common_names'] = attr['value'].tolist()

        return pd.Series(result)

    data_sr = pd.concat(
        [data_sr, data_sr.apply(_extract_sr_food_attributes, axis=1)],
        axis=1
    )

    return data_sr


if __name__ == '__main__':
    data_ff = get_foundation_food()
    exit()
    data_sr = get_legacy_food()

    # Construct food table.
    data = pd.concat(
        [data_ff, data_sr], ignore_index=True
    ).drop(columns=['publication_date'])
    data['food_category'] = data['food_category_id'].apply(
        lambda x: food_categories.loc[x, 'description'] \
            if x in food_categories.index else None
    )

    food_categories = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food_category.csv"
    ).set_index('id')
    data = data.drop(columns=[
        'food_category_id',
        'common_names',

    ])

    data = data[[
        'id',
        'data_type',
        'description'
    ]]

    print(data)
    data.to_csv('tests/kg_v2/outputs/fdc.csv')
