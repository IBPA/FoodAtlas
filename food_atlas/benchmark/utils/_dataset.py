import os
import pickle

import numpy as np
import pandas as pd

from ...data_processing.common_utils.knowledge_graph import KnowledgeGraph


def load_dict_cid_to_inchikey():
    with open("food_atlas/benchmark/utils/cid_inchi_key_dict.pkl", 'rb') as f:
        dict_cid_to_inchikey = pickle.load(f)

    return dict_cid_to_inchikey


def load_cocoa_and_garlic_in_foodmine():
    data_cocoa = pd.read_pickle(
        'food_atlas/benchmark/FoodMine/FoodMine_Notable_Files/fm_cocoa.pkl'
    )
    print(f"FoodMind - Cocoa: Total {len(data_cocoa)}")
    data_cocoa = data_cocoa.query("pubchem_id.notnull()").astype(
        {'pubchem_id': 'Int64'}
    )
    data_cocoa = data_cocoa[
        data_cocoa['chem_id'].apply(lambda x: type(x) == str)
    ]

    print(f"FoodMind - Cocoa: After cleaning {len(data_cocoa)}")

    data_garlic = pd.read_pickle(
        'food_atlas/benchmark/FoodMine/FoodMine_Notable_Files/fm_garlic.pkl'
    )
    print(f"FoodMind - Garlic: Total {len(data_garlic)}")
    data_garlic = data_garlic.query("pubchem_id.notnull()").astype(
        {'pubchem_id': 'Int64'}
    )
    data_garlic = data_garlic[
        data_garlic['chem_id'].apply(lambda x: type(x) == str)
    ]

    print(f"FoodMind - Garlic: After cleaning {len(data_garlic)}")

    return data_cocoa, data_garlic


def load_cocoa_and_garlic_in_foodatlas():
    PATH_OUTPUTS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        '..',
        'outputs',
        'benchmark',
    )

    if os.path.exists(f"{PATH_OUTPUTS_DIR}/data_cocoa.csv") and \
            os.path.exists(f"{PATH_OUTPUTS_DIR}/data_garlic.csv") and \
            os.path.exists(f"{PATH_OUTPUTS_DIR}/evidence_cocoa.csv") and \
            os.path.exists(f"{PATH_OUTPUTS_DIR}/evidence_garlic.csv"):
        data_cocoa = pd.read_csv(f"{PATH_OUTPUTS_DIR}/data_cocoa.csv")
        data_garlic = pd.read_csv(f"{PATH_OUTPUTS_DIR}/data_garlic.csv")
        evidence_cocoa = pd.read_csv(f"{PATH_OUTPUTS_DIR}/evidence_cocoa.csv")
        evidence_garlic = pd.read_csv(
            f"{PATH_OUTPUTS_DIR}/evidence_garlic.csv")
    else:
        kg = KnowledgeGraph('outputs/backend_data/v0.1')
        data = kg.get_kg()
        dict_cid_to_inchikey = load_dict_cid_to_inchikey()

        # Extract these foods from FoodAtlas.
        food_to_NCBI_taxonomy = {
            'cocoa': '3641',  # cocoa and its NCBI taxonomy
            'garlic': '4682',  # garlic and its NCBI taxonomy
        }

        print(
            "==================================================\n"
            "Extracting chemicals and IDs in FoodAtlas...\n"
            "=================================================="
        )

        data_food_dfs = []
        for food in food_to_NCBI_taxonomy.keys():
            print(f"Food: {food}")

            # Extract the targetted food in FoodAtlas.
            entities_food = kg.get_entity_by_other_db_id({
                'NCBI_taxonomy': [food_to_NCBI_taxonomy[food]],
            })['foodatlas_id'].tolist()
            data_food = data.query(
                f"head in {entities_food} & relation == 'r0'"
            )
            print(f"# of food entities: {len(entities_food)}")
            print(f"# of triples: {len(data_food)}")

            # Prepare the triples.
            # 1. Expand the chemical information to the triples.
            data_food = pd.concat(
                [
                    data_food.reset_index(drop=True),
                    pd.DataFrame(data_food['tail'].apply(
                        lambda x: kg.get_entity_by_id(x).to_dict()
                    ).tolist())
                ],
                axis=1,
            )
            # 2. Extract PubChem ID and MeSH ID from external IDs.
            data_food['pubchem_id'] = data_food['other_db_ids'].apply(
                lambda x: x['PubChem'][0] if 'PubChem' in x else np.nan
            ).astype('Int64')
            data_food['mesh_id'] = data_food['other_db_ids'].apply(
                lambda x: x['MESH'][0] if 'MESH' in x else np.nan
            )
            # 3. Drop triples without PubChem ID.
            data_food = data_food.query("pubchem_id.notnull()")
            print(f"# of triples with PubChem IDs: {len(data_food)}\n")
            # 4. Expand inchikey structure codes.
            data_food['inchikey'] = data_food['pubchem_id'].apply(
                lambda x: dict_cid_to_inchikey[x][0]
            )
            data_food['inchikey_structure_code'] = data_food['inchikey'].apply(
                lambda x: x.split('-')[0]
            )
            data_food_dfs += [data_food]

        print(
            "==================================================\n"
            "Extracting evidence from FoodAtlas...\n"
            "=================================================="
        )

        # Extract evidence for these foods.
        evidence = kg.get_evidence().set_index('triple')
        evidence['sort_index'] = evidence['tail'].apply(
            lambda x: int(x[1:])
        )

        # food_to_entity_food_whole = {
        #     'cocoa': 'e1314',
        #     'garlic': 'e1381',
        # }

        evidence_food_dfs = []
        for data_food, food in zip(
                data_food_dfs,
                ['cocoa', 'garlic']):
            print(f"Food: {food}")

            # Sort the triples by chemicals.
            data_food = data_food.sort_values(by=['tail'])
            print(f"# of unique chemicals: {data_food['tail'].nunique()}")

            # Extract the evidence for the triples.
            triples_food = data_food.apply(
                lambda row: f"({row['head']},{row['relation']},{row['tail']})",
                axis=1,
            ).tolist()
            evidence_food = evidence.loc[triples_food]
            if food == 'cocoa':
                evidence_food = evidence_food.query(
                    "hypothesis.str.contains('chocolate ') == False & "
                    "hypothesis.str.contains('Chocolate ') == False"
                )
            data_food_ = data_food.drop_duplicates(['tail']).set_index('tail')
            evidence_food['pubchem_id'] = evidence_food['tail'].apply(
                lambda x: data_food_.loc[x]['pubchem_id']
            )
            evidence_food['mesh_id'] = evidence_food['tail'].apply(
                lambda x: data_food_.loc[x]['mesh_id']
            )
            print(f"# of evidence: {len(evidence_food)}")

            # evidence_food.groupby(['premise', 'hypothesis']).apply(
            #     lambda group: print(group)
            #                   if len(group['mesh_id'].unique()) > 1
            #                   else None)

            evidence_food = pd.concat(
                [
                    evidence_food.query("mesh_id.notnull()").drop_duplicates(
                        subset=['premise', 'hypothesis', 'mesh_id']
                    ),
                    evidence_food.query("mesh_id.isnull()"),
                ],
            )
            print(
                f"# of evidence after dropping duplicates: "
                f"{len(evidence_food)}\n"
            )

            # Sort the evidence by chemicals and quality.
            evidence_food = evidence_food.sort_values(
                by=['sort_index', 'quality', 'prob_mean'],
                ascending=[True, False, False],
            )
            evidence_food_dfs += [evidence_food]

        data_cocoa, data_garlic = data_food_dfs
        evidence_cocoa, evidence_garlic = evidence_food_dfs
        data_cocoa.to_csv(f"{PATH_OUTPUTS_DIR}/data_cocoa.csv", index=False)
        data_garlic.to_csv(f"{PATH_OUTPUTS_DIR}/data_garlic.csv", index=False)
        evidence_cocoa.to_csv(
            f"{PATH_OUTPUTS_DIR}/evidence_cocoa.csv", index=False)
        evidence_garlic.to_csv(
            f"{PATH_OUTPUTS_DIR}/evidence_garlic.csv", index=False)

    return data_cocoa, data_garlic, evidence_cocoa, evidence_garlic
