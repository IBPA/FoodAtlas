from ast import literal_eval
from time import sleep
import os

import pandas as pd
from pandarallel import pandarallel
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

from ..data_processing.common_utils.knowledge_graph import KnowledgeGraph

pandarallel.initialize(progress_bar=True)


def load_inchikeys_to_cids():
    inchikeys_to_cids = pd.read_csv(
        "outputs/benchmark/inchikeys_to_cids_foodb.txt",
        sep='\t',
        header=None,
        names=['inchikey', 'cid'],
    ).astype({'cid': 'Int64'}).set_index('inchikey').to_dict()['cid']

    return inchikeys_to_cids


def load_foodb():
    # foods_foodb = pd.read_csv(
    #     "data/FooDB/foodb_2020_04_07_csv/Food.csv", low_memory=False
    # )
    # print(f"Original: {len(foods_foodb)}")
    # foods_foodb = foods_foodb.query("ncbi_taxonomy_id.notnull()")
    # foods_foodb = foods_foodb.astype({'ncbi_taxonomy_id': int})
    # foods_foodb = foods_foodb.set_index('id')
    # print(
    #     f"Unique NCBI Taxonomy ID: {foods_foodb['ncbi_taxonomy_id'].dropna().nunique()}"
    # )

    # chemicals_foodb = pd.read_csv(
    #     "data/FooDB/foodb_2020_04_07_csv/Compound.csv", low_memory=False)
    # print(f"Original: {len(chemicals_foodb)}")
    # chemicals_foodb = chemicals_foodb.rename(
    #     columns={'moldb_smiles': 'inchikey'})
    # chemicals_foodb = chemicals_foodb.query("inchikey.notnull()")
    # inchikeys_to_cids = load_inchikeys_to_cids()
    # chemicals_foodb['cid'] = chemicals_foodb['inchikey'].apply(
    #     lambda x: inchikeys_to_cids[x],
    # )
    # chemicals_foodb = chemicals_foodb.query("cid.notnull()")
    # chemicals_foodb = chemicals_foodb.set_index('id')
    # print(f"Unique PubChem CID: {chemicals_foodb['cid'].nunique()}")

    # data_foodb = pd.read_csv(
    #     "data/FooDB/foodb_2020_04_07_csv/Content.csv", low_memory=False
    # ).query("source_type == 'Compound'")
    # print(f"Original: {len(data_foodb)}")
    # data_foodb = data_foodb.query(
    #     f"food_id in {foods_foodb.index.tolist()} "
    #     f"& source_id in {chemicals_foodb.index.tolist()}"
    # )
    # # Add NCBI taxonomy IDs and CIDs.
    # data_foodb['ncbi_id'] = data_foodb['food_id'].parallel_apply(
    #     lambda x: foods_foodb.loc[x]['ncbi_taxonomy_id']
    # )
    # data_foodb['cid'] = data_foodb['source_id'].parallel_apply(
    #     lambda x: chemicals_foodb.loc[x]['cid']
    # )
    # print(f"Unique pairs: {len(data_foodb[['ncbi_id', 'cid']].drop_duplicates())}")

    if not os.path.exists("outputs/benchmark/foodb_for_foodb_comparison.csv"):
        # Only keep the foods with NCBI taxonomy IDs.
        foods_foodb = pd.read_csv(
            "data/FooDB/foodb_2020_04_07_csv/Food.csv", low_memory=False
        )
        foods_foodb = foods_foodb.query("ncbi_taxonomy_id.notnull()")
        foods_foodb = foods_foodb.astype({'ncbi_taxonomy_id': int})
        foods_foodb = foods_foodb.set_index('id')

        # Only keep the chemicals with PubChem CIDs.
        chemicals_foodb = pd.read_csv(
            "data/FooDB/foodb_2020_04_07_csv/Compound.csv", low_memory=False)
        chemicals_foodb = chemicals_foodb.rename(
            columns={'moldb_smiles': 'inchikey'})
        chemicals_foodb = chemicals_foodb.query("inchikey.notnull()")
        inchikeys_to_cids = load_inchikeys_to_cids()
        chemicals_foodb['cid'] = chemicals_foodb['inchikey'].apply(
            lambda x: inchikeys_to_cids[x],
        )
        chemicals_foodb = chemicals_foodb.query("cid.notnull()")
        chemicals_foodb = chemicals_foodb.set_index('id')

        data_foodb = pd.read_csv(
            "data/FooDB/foodb_2020_04_07_csv/Content.csv", low_memory=False
        ).query("source_type == 'Compound'")
        data_foodb = data_foodb.query(
            f"food_id in {foods_foodb.index.tolist()} "
            f"& source_id in {chemicals_foodb.index.tolist()}"
        )

        # Add NCBI taxonomy IDs and CIDs.
        data_foodb['ncbi_id'] = data_foodb['food_id'].parallel_apply(
            lambda x: foods_foodb.loc[x]['ncbi_taxonomy_id']
        )
        data_foodb['cid'] = data_foodb['source_id'].parallel_apply(
            lambda x: chemicals_foodb.loc[x]['cid']
        )

        data_foodb.to_csv(
            "outputs/benchmark/foodb_for_foodb_comparison.csv",
            index=False,
        )
    else:
        data_foodb = pd.read_csv(
            "outputs/benchmark/foodb_for_foodb_comparison.csv",
            low_memory=False,
        )

    return data_foodb


def load_foodatlas():
    if not os.path.exists(
            "outputs/benchmark/food_atlas_for_foodb_comparison.csv"):
        kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
        triplets = pd.read_csv("outputs/backend_data/v0.1/kg.txt", sep="\t")
        triplets = triplets.query("relation == 'r0'")

        def get_ncbi_and_pubchem_ids(row):
            head_id, tail_id = row['head'], row['tail']

            other_db_ids_head = kg.get_entity_by_id(head_id)['other_db_ids']
            ncbi_id = other_db_ids_head['NCBI_taxonomy'][0]

            other_db_ids_tail = kg.get_entity_by_id(tail_id)['other_db_ids']
            if 'PubChem' in other_db_ids_tail:
                cid = other_db_ids_tail['PubChem'][0]
            else:
                cid = None

            return pd.Series([ncbi_id, cid])

        triplets[['ncbi_id', 'cid']] = triplets.parallel_apply(
            lambda row: get_ncbi_and_pubchem_ids(row), axis=1
        )
        triplets = triplets.astype({'ncbi_id': 'Int64', 'cid': 'Int64'})
        triplets.to_csv(
            "outputs/benchmark/food_atlas_for_foodb_comparison.csv",
            index=False,
        )
    else:
        triplets = pd.read_csv(
            "outputs/benchmark/food_atlas_for_foodb_comparison.csv",
            low_memory=False,
        )
        triplets = triplets.astype({'ncbi_id': 'Int64', 'cid': 'Int64'})

    return triplets


if __name__ == '__main__':
    # data_food_atlas = load_foodatlas()
    data_foodb = load_foodb()

    # # Chemicals.
    # venn2(
    #     subsets=(
    #         set(data_food_atlas['cid'].dropna().tolist()),
    #         set(data_foodb['cid'].dropna().tolist())
    #     ),
    #     set_labels=('FoodAtlas', 'FooDB'),
    # )
    # plt.savefig('outputs/benchmark/venn_fa_foodb_chemicals.svg')
    # plt.close()

    # # Foods.
    # venn2(
    #     subsets=(
    #         set(data_food_atlas['ncbi_id'].dropna().tolist()),
    #         set(data_foodb['ncbi_id'].dropna().tolist())
    #     ),
    #     set_labels=('FoodAtlas', 'FooDB'),
    # )
    # plt.savefig('outputs/benchmark/venn_fa_foodb_foods.svg')
    # plt.close()

    # # Triplets.
    # venn2(
    #     subsets=(
    #         set(
    #             data_food_atlas[['ncbi_id', 'cid']].dropna().parallel_apply(
    #                 lambda x: (x['ncbi_id'], x['cid']), axis=1
    #             ).tolist()
    #         ),
    #         set(
    #             data_foodb[['ncbi_id', 'cid']].dropna().parallel_apply(
    #                 lambda x: (x['ncbi_id'], x['cid']), axis=1
    #             ).tolist()
    #         )
    #     ),
    #     set_labels=('FoodAtlas', 'FooDB'),
    # )
    # plt.savefig('outputs/benchmark/venn_fa_foodb_triplets.svg')
    # plt.close()

    # # Triplets with FooDB removing predicted triplets.
    # data_foodb = data_foodb.query("citation_type != 'PREDICTED'")
    # venn2(
    #     subsets=(
    #         set(
    #             data_food_atlas[['ncbi_id', 'cid']].dropna().parallel_apply(
    #                 lambda x: (x['ncbi_id'], x['cid']), axis=1
    #             ).tolist()
    #         ),
    #         set(
    #             data_foodb[['ncbi_id', 'cid']].dropna().parallel_apply(
    #                 lambda x: (x['ncbi_id'], x['cid']), axis=1
    #             ).tolist()
    #         )
    #     ),
    #     set_labels=('FoodAtlas', 'FooDB w/o Pathway Predictions)'),
    # )
    # plt.savefig('outputs/benchmark/venn_fa_foodb_no_pred_triplets.svg')
    # plt.close()

    # # Count triplet quality.
    # kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
    # triplets = pd.read_csv("outputs/backend_data/v0.1/kg.txt", sep="\t")
    # triplets = triplets.query("relation == 'r0'")
    # print(f"Original: {len(triplets)}")

    # def get_ncbi_and_pubchem_ids(row):
    #     head_id, tail_id = row['head'], row['tail']

    #     other_db_ids_head = kg.get_entity_by_id(head_id)['other_db_ids']
    #     ncbi_id = other_db_ids_head['NCBI_taxonomy'][0]

    #     other_db_ids_tail = kg.get_entity_by_id(tail_id)['other_db_ids']
    #     if 'PubChem' in other_db_ids_tail:
    #         cid = other_db_ids_tail['PubChem'][0]
    #     else:
    #         cid = None

    #     return pd.Series([ncbi_id, cid])

    # triplets[['ncbi_id', 'cid']] = triplets.parallel_apply(
    #     lambda row: get_ncbi_and_pubchem_ids(row), axis=1
    # )
    # triplets = triplets.astype({'ncbi_id': 'Int64', 'cid': 'Int64'})
    # print(f"Unique NCBI Taxonomy ID: {triplets['ncbi_id'].dropna().nunique()}")
    # print(f"Unique PubChem CID: {triplets['cid'].dropna().nunique()}")
    # print(
    #     f"Unique pairs: "
    #     f"{len(triplets[['ncbi_id', 'cid']].dropna().drop_duplicates())}"
    # )

    # Get triplet quality.
    # triplets = pd.read_csv(
    #     "outputs/benchmark/food_atlas_for_foodb_comparison.csv",
    #     low_memory=False,
    # )
    # triplets = triplets.astype({'ncbi_id': 'Int64', 'cid': 'Int64'})
    # print(triplets)

    # kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
    # evidence = kg.get_evidence().set_index('triple')
    # print(evidence)

    # def get_highest_quality(row, evidence):
    #     evidence_ = evidence.loc[[f"({row['head']},r0,{row['tail']})"]]
    #     qualities = evidence_['quality'].tolist()

    #     if 'high' in qualities:
    #         return 'high'
    #     elif 'medium' in qualities:
    #         return 'medium'
    #     elif 'low' in qualities:
    #         return 'low'
    #     else:
    #         raise ValueError

    # triplets['best_source_quality'] = triplets.parallel_apply(
    #     lambda row: get_highest_quality(row, evidence),
    #     axis=1,
    # )
    # triplets.to_csv(
    #     "outputs/benchmark/food_atlas_for_foodb_comparison_with_best_source.csv",
    #     index=False,
    # )

    triplets = pd.read_csv(
        "outputs/benchmark/"
        "food_atlas_for_foodb_comparison_with_best_source.csv",
        low_memory=False,
    )
    triplets = triplets.astype({'ncbi_id': 'Int64', 'cid': 'Int64'})
    triplets = triplets.dropna(subset=['ncbi_id', 'cid'])

    def get_best_quality_pairs(group, pairs):
        qualities = group['best_source_quality'].tolist()

        if 'high' in qualities:
            quality = 'high'
        elif 'medium' in qualities:
            quality = 'medium'
        elif 'low' in qualities:
            quality = 'low'

        pairs += [{
            'ncbi_id': group['ncbi_id'].iloc[0],
            'cid': group['cid'].iloc[0],
            'quality': quality,
        }]

    pairs_fa = []
    triplets.groupby(['ncbi_id', 'cid']).apply(
        lambda group: get_best_quality_pairs(group, pairs_fa)
    )
    pairs_fa = pd.DataFrame(pairs_fa)
    pairs_fa['pair_id'] = pairs_fa.apply(
        lambda x: f"{x['ncbi_id']}_{x['cid']}", axis=1
    )
    # pairs_fa = pairs_fa.set_index('pair_id')

    pairs_foodb = data_foodb[['ncbi_id', 'cid']].dropna().drop_duplicates()
    pairs_foodb['pair_id'] = pairs_foodb.apply(
        lambda x: f"{x['ncbi_id']}_{x['cid']}", axis=1
    )
    # pairs_foodb = pairs_foodb.set_index('pair_id')

    pairs_foodb_no_pw = data_foodb.query(
        "citation_type != 'PREDICTED'"
    )[['ncbi_id', 'cid']].dropna().drop_duplicates()
    pairs_foodb_no_pw['pair_id'] = pairs_foodb_no_pw.apply(
        lambda x: f"{x['ncbi_id']}_{x['cid']}", axis=1
    )
    # pairs_foodb_no_pw = pairs_foodb_no_pw.set_index('pair_id')

    pairs_fa_excl = pairs_fa.query(
        f"pair_id not in {pairs_foodb['pair_id'].tolist()}"
    )
    print(len(pairs_fa_excl))
    print(pairs_fa_excl['quality'].value_counts())
    # FoodAtlas not in FooDB:
    # High: 2091, Medium: 94095, Low: 9896

    pairs_fa_excl_2 = pairs_fa.query(
        f"pair_id not in {pairs_foodb_no_pw['pair_id'].tolist()}"
    )
    print(len(pairs_fa_excl_2))
    print(pairs_fa_excl_2['quality'].value_counts())
    # FoodAtlas not in FooDB (w/o pathway predictions):
    # High: 2091, Medium: 94138, Low: 9896

    # FoodAtlas:
    # High: 2653, Medium: 109682, Low: 13747
