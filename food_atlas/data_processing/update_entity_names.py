import argparse
import itertools
import os
import sys
import xml.etree.ElementTree as ET

from tqdm import tqdm
import pandas as pd

from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.knowledge_graph import CandidateEntity, CandidateRelation
from common_utils.utils import save_pkl, load_pkl
from common_utils.chemical_db_ids import read_mesh_data

MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"


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

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    #
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.input_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.input_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    df_chemicals = fa_kg.get_entities_by_type(exact_type="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = []
    for x in df_chemicals["other_db_ids"].tolist():
        if "MESH" in x:
            mesh_ids.extend(x["MESH"])
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    mesh_data_dict = read_mesh_data(args.use_pkl)
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

    # generate and add triple
    relation = CandidateRelation(
        name='isA',
        translation='is a',
    )

    triples = []
    for head_mesh_id, tail_mesh_id, in tqdm(head_tail_pairs):
        if head_mesh_id.startswith("C"):
            head_name = supp_mesh_id_name_lookup[head_mesh_id]
        elif head_mesh_id.startswith("D"):
            head_name = desc_mesh_id_name_lookup[head_mesh_id]
        head_ent = CandidateEntity(
            type="chemical",
            name=head_name,
            synonyms=[],
            other_db_ids={"MESH": [head_mesh_id]}
        )

        if tail_mesh_id.startswith("C"):
            tail_name = supp_mesh_id_name_lookup[tail_mesh_id]
        elif tail_mesh_id.startswith("D"):
            tail_name = desc_mesh_id_name_lookup[tail_mesh_id]

        tail_ent = CandidateEntity(
            type="chemical",
            name=tail_name,
            synonyms=[],
            other_db_ids={"MESH": [tail_mesh_id]}
        )

        triples.append([head_ent, relation, tail_ent])
    df_candidate_triples = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    df_candidate_triples["source"] = "MeSH"
    df_candidate_triples["quality"] = "high"
    fa_kg.add_triples(df_candidate_triples)

    # enter MeSH tree number and overwrite name
    df_chemicals = fa_kg.get_entities_by_type(exact_type="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = []
    for x in df_chemicals["other_db_ids"].tolist():
        if "MESH" in x:
            mesh_ids.extend(x["MESH"])
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    for mesh_id in tqdm(mesh_ids):
        df_entity = fa_kg.get_entity_by_other_db_id("MESH", mesh_id)
        assert df_entity.shape[0] == 1
        entity = df_entity.iloc[0]
        other_db_ids = entity["other_db_ids"]

        official_name = df_entity["name"]
        if mesh_id.startswith("C"):
            if mesh_id in supp_mesh_id_name_lookup:
                official_name = supp_mesh_id_name_lookup[mesh_id]

        if mesh_id.startswith("D"):
            if mesh_id in desc_mesh_id_name_lookup:
                official_name = desc_mesh_id_name_lookup[mesh_id]

            if mesh_id in desc_mesh_id_tree_number_lookup:
                mesh_tree = desc_mesh_id_tree_number_lookup[mesh_id]
                if "MESH_tree" in other_db_ids:
                    other_db_ids["MESH_tree"].extend(mesh_tree)
                else:
                    other_db_ids["MESH_tree"] = mesh_tree

        if other_db_ids == entity["other_db_ids"] and official_name is df_entity["name"]:
            continue

        fa_kg._overwrite_entity(
            foodatlas_id=entity["foodatlas_id"],
            name=official_name,
            type_=entity["type"],
            other_db_ids=other_db_ids,
        )

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.output_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
















import argparse
import os
import sys

from tqdm import tqdm
import pandas as pd

from common_utils.knowledge_graph import KnowledgeGraph, CandidateEntity, CandidateRelation


NODES_FILEPATH = "../../data/NCBI_Taxonomy/nodes.dmp"
NAMES_FILEPATH = "../../data/NCBI_Taxonomy/names.dmp"
TAXLINEAGE_FILEPATH = "../../data/NCBI_Taxonomy/taxidlineage.dmp"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"


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

    args = parser.parse_args()
    return args


def read_dmp_files(filepath: str, filetype: str):
    with open(filepath, "r") as _f:
        lines = _f.readlines()

    if filetype == "nodes":
        lines = [x.replace("\t|\n", "").split("\t|\t")[:3] for x in lines]
        columns = ["tax_id", "parent_tax_id", "rank"]
    elif filetype == "names":
        lines = [x.replace("\t|\n", "").split("\t|\t") for x in lines]
        columns = ["tax_id", "name_txt", "unique_name", "name_class"]
    elif filetype == "taxlineage":
        lines = [x.replace("\t|\n", "").split("\t|\t") for x in lines]
        columns = ["tax_id", "lineage"]
    else:
        raise ValueError()

    return pd.DataFrame(lines, columns=columns)


def get_matching_row(df: pd.DataFrame, tax_id: str):
    matching_node = df[df["tax_id"] == tax_id]
    if matching_node.shape[0] > 1:
        raise RuntimeError(f"More than one matching nodes: {tax_id}")
    elif matching_node.shape[0] == 0:
        return None
    else:
        return matching_node.iloc[0]


def get_names_dict(df_names: pd.DataFrame, tax_id: str):
    matching_name = df_names[df_names["tax_id"] == tax_id]
    names_dict = matching_name.groupby("name_class")["name_txt"].agg(list).reset_index()
    names_dict = dict(zip(names_dict["name_class"], names_dict["name_txt"]))
    assert "scientific name" in names_dict and len(names_dict["scientific name"]) == 1

    return names_dict


def main():
    args = parse_argument()

    # read KG
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.input_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.input_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    df_organisms = fa_kg.get_entities_by_type(exact_type="organism")
    print(f"Number of organisms in KG: {df_organisms.shape[0]}")

    df_organisms_with_part = fa_kg.get_entities_by_type(exact_type="organism_with_part")
    print(f"Number of organisms_with_part in KG: {df_organisms_with_part.shape[0]}")

    kg_tax_ids = [y for x in df_organisms["other_db_ids"].tolist() for y in x["NCBI_taxonomy"]]
    prev_len = len(kg_tax_ids)
    kg_tax_ids = list(set(kg_tax_ids))
    assert len(kg_tax_ids) == prev_len

    # nodes and names
    print("Reading nodes.dmp...")
    df_nodes = read_dmp_files(NODES_FILEPATH, filetype="nodes")
    print(f"Number of nodes: {df_nodes.shape[0]}")

    print("Reading names.dmp...")
    df_names = read_dmp_files(NAMES_FILEPATH, filetype="names")
    print(f"Number of names: {df_names.shape[0]}")

    # let's update existing nodes
    print("Updating entities...")

    # doing this will make it faster to run
    df_nodes_subset = df_nodes[df_nodes["tax_id"].apply(lambda x: x in kg_tax_ids)]
    df_names_subset = df_names[df_names["tax_id"].apply(lambda x: x in kg_tax_ids)]

    df_entities_to_update = pd.concat([df_organisms, df_organisms_with_part]).reset_index(drop=True)
    for idx, row in tqdm(df_entities_to_update.iterrows(), total=df_entities_to_update.shape[0]):
        query_id = row["other_db_ids"]["NCBI_taxonomy"]
        assert len(query_id) == 1
        query_id = query_id[0]

        matching_node = get_matching_row(df_nodes_subset, query_id)
        if matching_node is None:
            continue
        matching_name = df_names_subset[df_names_subset["tax_id"] == query_id]
        names_dict = matching_name.groupby("name_class")["name_txt"].agg(list).reset_index()
        names_dict = dict(zip(names_dict["name_class"], names_dict["name_txt"]))
        assert "scientific name" in names_dict and len(names_dict["scientific name"]) == 1

        if row["type"] == "organism":
            type_ = f"organism:{matching_node['rank']}"
        elif row["type"] == "organism_with_part":
            type_ = f"organism_with_part:{matching_node['rank']}"
        else:
            raise RuntimeError()

        fa_kg._overwrite_entity(
            foodatlas_id=row["foodatlas_id"],
            type_=type_,
        )

    # taxlineage
    print("Reading taxlineage.dmp...")
    df_taxlineage = read_dmp_files(TAXLINEAGE_FILEPATH, filetype="taxlineage")
    df_taxlineage["lineage"] = df_taxlineage["lineage"].apply(lambda x: x.strip().split(" "))
    print(f"Number of taxlineage: {df_taxlineage.shape[0]}")

    # get relevant entities to speed things up
    print("Generating candidate entities...")
    df_taxlineage_subset = df_taxlineage[df_taxlineage["tax_id"].apply(lambda x: x in kg_tax_ids)]
    print(f"Number of taxlineage subset: {df_taxlineage_subset.shape[0]}")

    parent_tax_ids = []
    all_lineage_tax_ids = []
    all_lineage_pairs = []
    for tax_id in tqdm(kg_tax_ids):
        matching_taxlineage = get_matching_row(df_taxlineage_subset, tax_id)
        if matching_taxlineage is None:
            continue

        lineage = matching_taxlineage["lineage"] + [tax_id]
        parent_tax_ids.append(lineage[0])

        all_lineage_tax_ids.extend(lineage)
        for idx in range(len(lineage)-1):
            all_lineage_pairs.append((lineage[idx], lineage[idx+1]))

    parent_tax_ids = list(set(parent_tax_ids))
    print(f"Parent tax IDs: {parent_tax_ids}")

    all_lineage_tax_ids = list(set(all_lineage_tax_ids))
    all_lineage_pairs = list(set(all_lineage_pairs))
    print(f"Number of candidate entities to add from NCBI taxonomy: {len(all_lineage_tax_ids)}")
    print(f"Number of candidate pairs to add from NCBI taxonomy: {len(all_lineage_pairs)}")

    print("Gettiing subset of nodes and names...")
    df_nodes_subset = df_nodes[df_nodes["tax_id"].apply(lambda x: x in all_lineage_tax_ids)]
    df_names_subset = df_names[df_names["tax_id"].apply(lambda x: x in all_lineage_tax_ids)]

    print("Generating candidate entity dictionary...")
    candidate_entity_dict = {}
    for tax_id in tqdm(all_lineage_tax_ids):
        matching_node = get_matching_row(df_nodes_subset, tax_id)
        if matching_node is None:
            raise ValueError()
        names_dict = get_names_dict(df_names_subset, tax_id)

        ent = CandidateEntity(
            type=f"organism:{matching_node['rank']}",
            name=names_dict["scientific name"][0],
            synonyms=[x for k, v in names_dict.items() if k != "scientific name" for x in v],
            other_db_ids={"NCBI_taxonomy": [tax_id]}
        )

        candidate_entity_dict[tax_id] = ent

    # now add knowledge graph
    print("Generating candidate triples...")

    relation = CandidateRelation(
        name='hasChild',
        translation='has child',
    )

    data = []
    for head_tax_id, tail_tax_id in tqdm(all_lineage_pairs):
        data.append(
            [candidate_entity_dict[head_tax_id], relation, candidate_entity_dict[tail_tax_id]])
    df_candiate_triples = pd.DataFrame(data, columns=["head", "relation", "tail"])
    df_candiate_triples["source"] = "NCBI_taxonomy"
    df_candiate_triples["quality"] = "high"
    fa_kg.add_triples(df_candiate_triples)

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.output_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
