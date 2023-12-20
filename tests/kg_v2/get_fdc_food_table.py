import pandas as pd

ATTR_NAMES = [
    'FoodOn Ontology name for FDC item',
    'FoodOn Ontology ID for FDC item',
    'NCBI Taxon',
]


if __name__ == '__main__':
    food_categories = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food_category.csv"
    ).set_index('id')

    # Load foundation food.
    data_ff = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_foundation_food_csv_2023-10-26/"
        "food.csv",
    ).query("data_type == 'foundation_food'")
    data_ff_attr = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_foundation_food_csv_2023-10-26/"
        "food_attribute.csv",
    ).set_index('fdc_id')
    data_ff_attr_id = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_foundation_food_csv_2023-10-26/"
        "food_attribute_type.csv",
    )

    def _extract_ff_food_attributes(row):
        result = {
            'common_names': [],
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
                ATTR_NAMES,
                ['foodon_name', 'foodon_url', 'ncbi_taxon_url']
            ):
                result[col_name] = attr.loc[attr_name, 'value'] \
                    if attr_name in attr.index else None

        return pd.Series(result)

    data_ff = pd.concat(
        [data_ff, data_ff.apply(_extract_ff_food_attributes, axis=1)],
        axis=1
    )

    # # Load legacy food.
    data_sr = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food.csv",
    )
    data_sr_attr = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food_attribute.csv",
    ).set_index('fdc_id')
    data_sr_attr_id = pd.read_csv(
        "tests/kg_v2/data/FDC/FoodData_Central_sr_legacy_food_csv_2018-04/"
        "food_attribute_type.csv",
    )

    def _extract_sr_food_attributes(row):
        result = {
            'common_names': [],
            'foodon_name': None,
            'foodon_url': None,
            'ncbi_taxon_url': None,
        }
        if row['fdc_id'] in data_sr_attr.index:
            attr = data_sr_attr.loc[row['fdc_id']]
            print(attr)
        #     if type(attr) == pd.Series:
        #         attr = pd.DataFrame(attr).T
        #     attr = attr.set_index('name')

        #     for attr_name, col_name in zip(
        #         ATTR_NAMES,
        #         ['foodon_name', 'foodon_url', 'ncbi_taxon_url']
        #     ):
        #         result[col_name] = attr.loc[attr_name, 'value'] \
        #             if attr_name in attr.index else None

        # return pd.Series(result)

    print(data_sr.apply(_extract_sr_food_attributes, axis=1))
    # print(data_sr)
    # print(data_sr_attr)
    # print(data_sr_attr_id)
    # # Construct food table.
    # data = pd.concat(
    #     [data_ff, data_sr], ignore_index=True
    # ).drop(columns=['publication_date'])
    # data['food_category'] = data['food_category_id'].apply(
    #     lambda x: food_categories.loc[x, 'description'] \
    #         if x in food_categories.index else None
    # )

    # print(data)



    pass
