import argparse
import math
import os
import sys

import pandas as pd

from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.utils import load_pkl


MESH_DATA_DIR = "../../data/MESH"
FOODB_FOOD_FILEPATH = "../../data/FooDB/foodb_2020_04_07_csv/Food.csv"
ORGANISMS_GROUP_FILENAME = "organisms_group.txt"
CHEMICALS_GROUP_FILENAME = "chemicals_group.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="Final KG dir",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # read KG and save all files to the new folder
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    fa_kg.save(kg_dir=args.output_dir)

    # load entities
    df_organisms = fa_kg.get_entities_by_type(startswith_type="organism")
    for x in df_organisms["other_db_ids"].tolist():
        assert "NCBI_taxonomy" in x
        assert len(x["NCBI_taxonomy"]) == 1
    df_organisms["NCBI_taxonomy"] = df_organisms["other_db_ids"].apply(
        lambda x: x["NCBI_taxonomy"][0])
    df_organisms = df_organisms[["foodatlas_id", "NCBI_taxonomy"]]

    df_chemicals = fa_kg.get_entities_by_type(startswith_type="chemical")
    df_chemicals["MESH"] = df_chemicals["other_db_ids"].apply(
        lambda x: x["MESH"] if "MESH" in x else [])
    df_chemicals = df_chemicals.explode("MESH")

    # we need to find foods and chemicals that has contains relation
    # not the taxonomic/ontoligical ones
    contains_foodatlas_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_evidence = fa_kg.get_evidence()
    df_evidence_contains = df_evidence[df_evidence["relation"] == contains_foodatlas_id]
    food_chem_entities = \
        df_evidence_contains["head"].tolist() + df_evidence_contains["tail"].tolist()
    food_chem_entities = list(set(food_chem_entities))

    df_organisms["is_food"] = df_organisms["foodatlas_id"].apply(
        lambda x: x in food_chem_entities)
    df_chemicals["is_contains"] = df_chemicals["foodatlas_id"].apply(
        lambda x: x in food_chem_entities)

    # find group and subgroup for foods
    df_foodb_food = pd.read_csv(FOODB_FOOD_FILEPATH, dtype=str)
    df_foodb_food.dropna(subset="ncbi_taxonomy_id", inplace=True)
    ncbi_group_map = dict(zip(
        df_foodb_food["ncbi_taxonomy_id"].tolist(), df_foodb_food["food_group"].tolist()
    ))
    ncbi_subgroup_map = dict(zip(
        df_foodb_food["ncbi_taxonomy_id"].tolist(), df_foodb_food["food_subgroup"].tolist()
    ))

    df_organisms["group"] = df_organisms["NCBI_taxonomy"].apply(
        lambda x: ncbi_group_map[x] if x in ncbi_group_map else "")
    df_organisms["subgroup"] = df_organisms["NCBI_taxonomy"].apply(
        lambda x: ncbi_subgroup_map[x] if x in ncbi_subgroup_map else "")
    df_organisms = df_organisms[["foodatlas_id", "is_food", "group", "subgroup"]]
    df_organisms.to_csv(
        os.path.join(args.output_dir, ORGANISMS_GROUP_FILENAME),
        sep='\t',
        index=False,
    )

    # find group and subgroup for chemicals
    desc_mesh_id_tree_number_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_tree_number_lookup.pkl')
    desc_tree_number_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_tree_number_name_lookup.pkl')
    supp_mesh_id_heading_mesh_id_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'supp_mesh_id_heading_mesh_id_lookup.pkl')

    desc_mesh_id_tree_number_lookup_filepath = load_pkl(desc_mesh_id_tree_number_lookup_filepath)
    desc_tree_number_name_lookup_filepath = load_pkl(desc_tree_number_name_lookup_filepath)
    supp_mesh_id_heading_mesh_id_lookup = load_pkl(supp_mesh_id_heading_mesh_id_lookup_filepath)

    def _get_mesh_tree(x):
        if type(x) != str and math.isnan(x):
            return ""

        if x.startswith("D"):
            if x not in desc_mesh_id_tree_number_lookup_filepath:
                return ""
            x_tree_numbers = desc_mesh_id_tree_number_lookup_filepath[x]
        elif x.startswith("C"):
            x_tree_numbers = []
            if x not in supp_mesh_id_heading_mesh_id_lookup:
                return ""
            heading_mesh_ids = supp_mesh_id_heading_mesh_id_lookup[x]
            for heading_mesh_id in heading_mesh_ids:
                x_tree_numbers.extend(desc_mesh_id_tree_number_lookup_filepath[heading_mesh_id])
        else:
            raise RuntimeError()

        assert len(x_tree_numbers) != 0
        x_tree_numbers = list(set(['.'.join(n.split('.')[:2]) for n in x_tree_numbers]))
        return x_tree_numbers

    def _get_group_or_subgroup(tree_number, group_or_subgroup):
        if tree_number == '':
            return "Unknown"

        if group_or_subgroup == "group":
            group_tree_number = tree_number.split('.')[0]
            return desc_tree_number_name_lookup_filepath[group_tree_number]
        elif group_or_subgroup == "subgroup":
            if tree_number.count('.') == 0:
                return "None"
            assert tree_number.count('.') == 1
            return desc_tree_number_name_lookup_filepath[tree_number]
        else:
            raise NotImplementedError()

    df_chemicals["MESH_tree"] = df_chemicals["MESH"].apply(lambda x: _get_mesh_tree(x))
    df_chemicals = df_chemicals.explode("MESH_tree")
    df_chemicals["group"] = df_chemicals["MESH_tree"].apply(
        lambda x: _get_group_or_subgroup(x, "group"))
    df_chemicals["subgroup"] = df_chemicals["MESH_tree"].apply(
        lambda x: _get_group_or_subgroup(x, "subgroup"))
    df_chemicals = df_chemicals[["foodatlas_id", "is_contains", "group", "subgroup"]]
    df_chemicals.to_csv(
        os.path.join(args.output_dir, CHEMICALS_GROUP_FILENAME),
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    main()
