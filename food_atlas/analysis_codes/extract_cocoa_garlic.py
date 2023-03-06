import sys
sys.path.append('../data_processing/')

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

# change this to your directory
fa_kg = KnowledgeGraph('../../outputs/backend_data/v0.1')

# this is all KG triples
df_kg = fa_kg.get_kg()
print(df_kg)

# let's find the foodatlas_id for 'contains' relation type
contains_foodatlas_id = fa_kg.get_relation_by_name('contains')['foodatlas_id']
print(f'FoodAtlas ID for contains relation: {contains_foodatlas_id}')

# let's iterate over these two foods in foodmine
two_foods = {
    'cocoa': '3641',  # cocoa and its NCBI taxonomy
    'garlic': '4682',  # garlic and its NCBI taxonomy
}

for food_name, ncbi_taxonomy in two_foods.items():
    # if we search by specific part ID (p0 = whole food), we get one entity
    food_whole = fa_kg.get_entity_by_other_db_id({
        'NCBI_taxonomy': [ncbi_taxonomy],
        'foodatlas_part_id': 'p0',
    })
    print(food_whole)
    food_whole_entity = food_whole['foodatlas_id'].tolist()

    # if we omit the part ID, we get multiple food entities (whole and parts)
    food_all = fa_kg.get_entity_by_other_db_id({
        'NCBI_taxonomy': ['3641'],
        # 'foodatlas_part_id': 'p0',
    })
    print(food_all)
    food_all_entity = food_all['foodatlas_id'].tolist()

    # triples with whole food as the head entity
    df_kg_food_whole = df_kg[df_kg['head'].apply(lambda x: x in food_whole_entity)]
    print(df_kg_food_whole)
    print(f'KG size of (whole food, contains, chemical): {df_kg_food_whole.shape[0]}')
    chemicals_food_whole = sorted(set(df_kg_food_whole['tail'].tolist()))

    print()
    print('These are the chemicals that are in whole food')
    for idx, c in enumerate(chemicals_food_whole):
        entity = fa_kg.get_entity_by_id(c)
        print(f'{idx+1}: {entity.to_dict()}')

    # triples with cacao whole food and food parts as the head entity
    df_kg_food_all = df_kg[df_kg['head'].apply(lambda x: x in food_all_entity)]
    print(df_kg_food_all)
    print(f'KG size of (whole food and food parts, contains, chemical): {df_kg_food_all.shape[0]}')
    chemicals_food_all = sorted(set(df_kg_food_all['tail'].tolist()))

    print()
    print('These are the chemicals that are in whole food and food parts')
    for idx, c in enumerate(chemicals_food_all):
        entity = fa_kg.get_entity_by_id(c)
        print(f'{idx+1}: {entity.to_dict()}')

    print()
    print()
    print()
