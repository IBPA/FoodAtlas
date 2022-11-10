import argparse
import os
import sys
import xml.etree.ElementTree as ET

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph, CandidateEntity, CandidateRelation  # noqa: E402


DESC_FILEPATH = "../../data/MESH/desc2022_small.xml"
# DESC_FILEPATH = "../../data/MESH/desc2022.xml"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    #
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.kg_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.kg_dir, RELATIONS_FILENAME),
    )

    df_chemicals = fa_kg.get_entities_by_type(type_="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = [x["MESH"] for x in df_chemicals["other_db_ids"].tolist()]
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    #
    print("Loading descriptor XML...")
    desc = ET.parse(DESC_FILEPATH)
    root = desc.getroot()

    print("Generating parent map...")
    parent_map = {c: p for p in root.iter() for c in p}

    print("Generating mesh_id_element_lookup...")
    mesh_id_element_lookup = {}
    mesh_id_tree_number_lookup = {}
    for x in root.iter("DescriptorUI"):
        if parent_map[x].tag == "DescriptorRecord":
            mesh_id_element_lookup[x.text] = x

    mesh_id_tree_number_lookup = {}
    for x in root.iter("TreeNumber"):
        print(x.text)
        if parent_map[x].tag == "TreeNumberList":
            mesh_id_element_lookup[x.text] = x

    # for mesh_id in mesh_ids:
    #     print(mesh_id)


if __name__ == '__main__':
    main()
