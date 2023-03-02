import argparse
from copy import deepcopy
import itertools
import os
import sys

from tqdm import tqdm
import pandas as pd

from common_utils.utils import read_dataframe
from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.knowledge_graph import CandidateEntity, CandidateRelation
from common_utils.chemical_db_ids import read_mesh_data, get_mesh_name_using_mesh_id
from common_utils.chemical_db_ids import read_pubchem_cid_mesh
from common_utils.chemical_db_ids import get_pubchem_id_data_dict_using

MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--output_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--use_pkl",
        action="store_true",
        help="Set if using pickled data.",
    )

    parser.add_argument(
        "--new_pkl",
        action="store_true",
        help="Set if using new pickled data.",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        help="Number of workers for pandarallel.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    #
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir, nb_workers=args.nb_workers)
    df_chemicals = fa_kg.get_entities_by_type(exact_type="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = []
    for x in df_chemicals["other_db_ids"].tolist():
        if "MESH" in x:
            mesh_ids.extend(x["MESH"])
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    mesh_data_dict = read_mesh_data()
    desc_mesh_id_tree_number_lookup = mesh_data_dict["desc_mesh_id_tree_number_lookup"]
    desc_mesh_id_name_lookup = mesh_data_dict["desc_mesh_id_name_lookup"]
    desc_tree_number_mesh_id_lookup = mesh_data_dict["desc_tree_number_mesh_id_lookup"]
    supp_mesh_id_heading_mesh_id_lookup = mesh_data_dict["supp_mesh_id_heading_mesh_id_lookup"]
    supp_mesh_id_name_lookup = mesh_data_dict["supp_mesh_id_name_lookup"]

    #
    print("Processing MESH IDs")

    def _descriptor_mesh_id_to_pairs(descriptor_mesh_id):
        pairs = []
        for tree_number in desc_mesh_id_tree_number_lookup[descriptor_mesh_id]:
            lineage = tree_number.split('.')
            for idx in range(len(lineage)-1):
                tail_mesh_id = desc_tree_number_mesh_id_lookup['.'.join(lineage[:idx+1])]
                head_mesh_id = desc_tree_number_mesh_id_lookup['.'.join(lineage[:idx+2])]
                pairs.append([head_mesh_id, tail_mesh_id])
        return pairs

    head_tail_pairs = []
    for mesh_id in tqdm(mesh_ids):
        if mesh_id.startswith('C'):
            if mesh_id not in supp_mesh_id_heading_mesh_id_lookup:
                continue
            for heading_mesh_id in supp_mesh_id_heading_mesh_id_lookup[mesh_id]:
                head_tail_pairs.append([mesh_id, heading_mesh_id])
                head_tail_pairs.extend(_descriptor_mesh_id_to_pairs(heading_mesh_id))
        elif mesh_id.startswith('D'):
            if mesh_id not in desc_mesh_id_tree_number_lookup:
                continue
            head_tail_pairs.extend(_descriptor_mesh_id_to_pairs(mesh_id))
        else:
            raise ValueError()

    # drop duplicates
    head_tail_pairs.sort()
    head_tail_pairs = list(k for k, _ in itertools.groupby(head_tail_pairs))
    print(f"Number of head tail pairs: {len(head_tail_pairs)}")

    mesh_ids = list(set([y for x in head_tail_pairs for y in x]))
    print(f"Found {len(mesh_ids)} unique MeSH IDs")
    print(f"Generating pubchem_id_data_dict...")
    pubchem_id_data_dict = get_pubchem_id_data_dict_using(
        mesh_ids, ['CAS', 'MESH', 'synonyms'], 'MESH')

    # generate and add triple
    relation = CandidateRelation(
        name='isA',
        translation='is a',
    )

    df_cid_mesh = read_pubchem_cid_mesh()

    def _get_other_db_ids_list(mesh_id):
        other_db_ids_list = []
        mesh_name = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
        if mesh_name is None:
            other_db_ids_list.append({"MESH": [mesh_id]})
        else:
            df_cid_mesh_match = df_cid_mesh[
                df_cid_mesh["mesh_name"].apply(lambda x: x == mesh_name)]
            pubchem_ids = list(set(df_cid_mesh_match["pubchem_id"].tolist()))
            if len(pubchem_ids) != 0:
                for pubchem_id in pubchem_ids:
                    other_db_ids = pubchem_id_data_dict[pubchem_id]
                    other_db_ids['PubChem'] = [pubchem_id]
                    other_db_ids_list.append(other_db_ids)
            else:
                other_db_ids_list.append({"MESH": [mesh_id]})

        return other_db_ids_list

    def _clean_other_db_ids(other_db_ids):
        return {
            k: v for k, v in other_db_ids.items()
            if k in ["MESH", "PubChem", "CAS"] and len(v) != 0
        }

    triples = []
    for head_mesh_id, tail_mesh_id, in tqdm(head_tail_pairs):
        if head_mesh_id.startswith("C"):
            head_name = supp_mesh_id_name_lookup[head_mesh_id]
        elif head_mesh_id.startswith("D"):
            head_name = desc_mesh_id_name_lookup[head_mesh_id]

        if tail_mesh_id.startswith("C"):
            tail_name = supp_mesh_id_name_lookup[tail_mesh_id]
        elif tail_mesh_id.startswith("D"):
            tail_name = desc_mesh_id_name_lookup[tail_mesh_id]

        head_other_db_ids_list = _get_other_db_ids_list(head_mesh_id)
        tail_other_db_ids_list = _get_other_db_ids_list(tail_mesh_id)
        head_tail_products = list(itertools.product(
            head_other_db_ids_list, tail_other_db_ids_list))

        for head_other_db_ids, tail_other_db_ids in head_tail_products:
            head_ent = CandidateEntity(
                type="chemical",
                name=head_name,
                synonyms=[],
                other_db_ids=_clean_other_db_ids(head_other_db_ids),
            )

            tail_ent = CandidateEntity(
                type="chemical",
                name=tail_name,
                synonyms=[],
                other_db_ids=_clean_other_db_ids(tail_other_db_ids),
            )

            triples.append([head_ent, relation, tail_ent])

    df_candidate_triples = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    df_candidate_triples["source"] = "MeSH"
    df_candidate_triples["quality"] = "medium"

    fa_kg.add_triples(df_candidate_triples)

    # enter MeSH tree number
    df_chemicals = fa_kg.get_entities_by_type(exact_type="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = []
    for x in df_chemicals["other_db_ids"].tolist():
        if "MESH" in x:
            mesh_ids.extend(x["MESH"])
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    print("Adding MeSH tree...")
    for mesh_id in tqdm(mesh_ids):
        df_entities = fa_kg.get_entity_by_other_db_id({"MESH": mesh_id})
        for _, row in df_entities.iterrows():
            other_db_ids = deepcopy(row["other_db_ids"])

            if mesh_id.startswith("D"):
                if mesh_id in desc_mesh_id_tree_number_lookup:
                    mesh_tree = desc_mesh_id_tree_number_lookup[mesh_id]
                    if "MESH_tree" in other_db_ids:
                        other_db_ids["MESH_tree"].extend(mesh_tree)
                    else:
                        other_db_ids["MESH_tree"] = mesh_tree

            if other_db_ids == row["other_db_ids"]:
                continue

            fa_kg._overwrite_entity(
                foodatlas_id=row["foodatlas_id"],
                type_=row["type"],
                other_db_ids=other_db_ids,
            )

    fa_kg.save(kg_dir=args.output_kg_dir)


if __name__ == '__main__':
    main()
