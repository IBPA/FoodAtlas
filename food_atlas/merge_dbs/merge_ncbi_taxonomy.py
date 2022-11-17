import argparse
import os
import sys

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph, CandidateEntity, CandidateRelation  # noqa: E402


NODES_FILEPATH = "../../data/NCBI_Taxonomy/nodes.dmp"
NAMES_FILEPATH = "../../data/NCBI_Taxonomy/names.dmp"
TAXLINEAGE_FILEPATH = "../../data/NCBI_Taxonomy/taxidlineage.dmp"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
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
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    df_organisms = fa_kg.get_entities_by_type(type_="organism")
    print(f"Number of organisms in KG: {df_organisms.shape[0]}")

    df_organisms_with_part = fa_kg.get_entities_by_type(type_="organism_with_part")
    print(f"Number of organisms_with_part in KG: {df_organisms_with_part.shape[0]}")

    kg_tax_ids = [x["NCBI_taxonomy"] for x in df_organisms["other_db_ids"].tolist()]
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

        fa_kg._update_entity(
            foodatlas_id=row["foodatlas_id"],
            type_=type_,
            name=names_dict["scientific name"][0],
            synonyms=[x for k, v in names_dict.items() if k != "scientific name" for x in v],
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

    all_lineage_tax_ids = []
    all_lineage_pairs = []
    for tax_id in tqdm(kg_tax_ids):
        matching_taxlineage = get_matching_row(df_taxlineage_subset, tax_id)
        if matching_taxlineage is None:
            continue

        lineage = matching_taxlineage["lineage"] + [tax_id]
        all_lineage_tax_ids.extend(lineage)
        for idx in range(len(lineage)-1):
            all_lineage_pairs.append((lineage[idx], lineage[idx+1]))

    all_lineage_tax_ids = list(set(all_lineage_tax_ids))
    all_lineage_pairs = list(set(all_lineage_pairs))
    print(f"Number of candidate entities to add from NCBI taxonomy: {len(all_lineage_tax_ids)}")
    print(f"Number of candidate pairs to add from NCBI taxonomy: {len(all_lineage_pairs)}")

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
            other_db_ids={"NCBI_taxonomy": tax_id}
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
    fa_kg.add_triples(df_candiate_triples, origin="NCBI_taxonomy")

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
