import argparse
import itertools
import os
import sys
import xml.etree.ElementTree as ET

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.knowledge_graph import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"
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
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    df_chemicals = fa_kg.get_entities_by_type(type_="chemical")
    print(f"Number of chemicals in KG: {df_chemicals.shape[0]}")

    mesh_ids = [x["MESH"] for x in df_chemicals["other_db_ids"].tolist()]
    mesh_ids = list(set(mesh_ids))
    print(f"Number of unique MESH ids: {len(mesh_ids)}")

    # parse descriptors
    desc_mesh_id_element_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_element_lookup.pkl')
    desc_mesh_id_tree_number_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_tree_number_lookup.pkl')
    desc_mesh_id_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_name_lookup.pkl')
    desc_tree_number_mesh_id_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_tree_number_mesh_id_lookup.pkl')
    desc_tree_number_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_tree_number_name_lookup.pkl')

    if args.use_pkl:
        desc_mesh_id_element_lookup = load_pkl(desc_mesh_id_element_lookup_filepath)
        desc_mesh_id_tree_number_lookup = load_pkl(desc_mesh_id_tree_number_lookup_filepath)
        desc_mesh_id_name_lookup = load_pkl(desc_mesh_id_name_lookup_filepath)
        desc_tree_number_mesh_id_lookup = load_pkl(desc_tree_number_mesh_id_lookup_filepath)
        desc_tree_number_name_lookup = load_pkl(desc_tree_number_name_lookup_filepath)
    else:
        print("Loading descriptor XML...")
        desc = ET.parse(DESC_FILEPATH)
        desc_root = desc.getroot()

        print("Generating descriptor parent map...")
        desc_parent_map = {c: p for p in desc_root.iter() for c in p}

        print("Generating descriptor lookups...")
        desc_mesh_id_element_lookup = {}
        desc_mesh_id_tree_number_lookup = {}
        desc_mesh_id_name_lookup = {}
        for x in desc_root.iter("DescriptorUI"):
            parent = desc_parent_map[x]
            if parent.tag != "DescriptorRecord":
                continue

            desc_mesh_id_element_lookup[x.text] = x

            for y in parent.iter("String"):
                if desc_parent_map[desc_parent_map[y]].tag != "DescriptorRecord":
                    continue
                desc_mesh_id_name_lookup[x.text] = y.text

            desc_mesh_id_tree_number_lookup[x.text] = []
            for y in parent.iter("TreeNumber"):
                assert desc_parent_map[y].tag == "TreeNumberList"
                desc_mesh_id_tree_number_lookup[x.text].append(y.text)

        desc_tree_number_mesh_id_lookup = {}
        for k, v in desc_mesh_id_tree_number_lookup.items():
            for x in v:
                desc_tree_number_mesh_id_lookup[x] = k

        desc_tree_number_name_lookup = {}
        for k, v in desc_tree_number_mesh_id_lookup.items():
            desc_tree_number_name_lookup[k] = desc_mesh_id_name_lookup[v]

        save_pkl(desc_mesh_id_element_lookup, desc_mesh_id_element_lookup_filepath)
        save_pkl(desc_mesh_id_tree_number_lookup, desc_mesh_id_tree_number_lookup_filepath)
        save_pkl(desc_mesh_id_name_lookup, desc_mesh_id_name_lookup_filepath)
        save_pkl(desc_tree_number_mesh_id_lookup, desc_tree_number_mesh_id_lookup_filepath)
        save_pkl(desc_tree_number_name_lookup, desc_tree_number_name_lookup_filepath)

    # parse supplementary
    supp_mesh_id_element_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'supp_mesh_id_element_lookup.pkl')
    supp_mesh_id_heading_mesh_id_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'supp_mesh_id_heading_mesh_id_lookup.pkl')
    supp_mesh_id_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'supp_mesh_id_name_lookup.pkl')

    if args.use_pkl:
        supp_mesh_id_element_lookup = load_pkl(supp_mesh_id_element_lookup_filepath)
        supp_mesh_id_heading_mesh_id_lookup = load_pkl(supp_mesh_id_heading_mesh_id_lookup_filepath)
        supp_mesh_id_name_lookup = load_pkl(supp_mesh_id_name_lookup_filepath)
    else:
        print("Loading supplementary XML...")
        supp = ET.parse(SUPP_FILEPATH)
        supp_root = supp.getroot()

        print("Generating supplementary parent map...")
        supp_parent_map = {c: p for p in supp_root.iter() for c in p}

        print("Generating supplementary lookups...")
        supp_mesh_id_element_lookup = {}
        supp_mesh_id_heading_mesh_id_lookup = {}
        supp_mesh_id_name_lookup = {}
        for x in supp_root.iter("SupplementalRecordUI"):
            parent = supp_parent_map[x]
            if parent.tag != "SupplementalRecord":
                continue

            supp_mesh_id_element_lookup[x.text] = x

            for y in parent.iter("String"):
                if supp_parent_map[supp_parent_map[y]].tag != "SupplementalRecord":
                    continue
                supp_mesh_id_name_lookup[x.text] = y.text

            supp_mesh_id_heading_mesh_id_lookup[x.text] = []
            for y in parent.iter("DescriptorUI"):
                if supp_parent_map[supp_parent_map[y]].tag != "HeadingMappedTo":
                    continue
                supp_mesh_id_heading_mesh_id_lookup[x.text].append(y.text.lstrip('*'))

        save_pkl(supp_mesh_id_element_lookup, supp_mesh_id_element_lookup_filepath)
        save_pkl(supp_mesh_id_heading_mesh_id_lookup, supp_mesh_id_heading_mesh_id_lookup_filepath)
        save_pkl(supp_mesh_id_name_lookup, supp_mesh_id_name_lookup_filepath)

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

    # update existing entities
    mesh_ids_to_add = [x for pairs in head_tail_pairs for x in pairs]
    mesh_ids_to_update = list(set(mesh_ids).intersection(mesh_ids_to_add))
    print("Updating existing entities...")
    for mesh_id in tqdm(mesh_ids_to_update):
        df_match = df_chemicals[df_chemicals["other_db_ids"].apply(lambda x: x["MESH"] == mesh_id)]
        assert df_match.shape[0] == 1

        if mesh_id.startswith("C"):
            name = supp_mesh_id_name_lookup[mesh_id]
        elif mesh_id.startswith("D"):
            name = desc_mesh_id_name_lookup[mesh_id]

        fa_kg._update_entity(
            foodatlas_id=df_match.iloc[0]["foodatlas_id"],
            type_="chemical",
            name=name,
            synonyms=[],
        )

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
            other_db_ids={"MESH": head_mesh_id}
        )

        if tail_mesh_id.startswith("C"):
            tail_name = supp_mesh_id_name_lookup[tail_mesh_id]
        elif tail_mesh_id.startswith("D"):
            tail_name = desc_mesh_id_name_lookup[tail_mesh_id]

        tail_ent = CandidateEntity(
            type="chemical",
            name=tail_name,
            synonyms=[],
            other_db_ids={"MESH": tail_mesh_id}
        )

        triples.append([head_ent, relation, tail_ent])
    df_candiate_triples = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    fa_kg.add_triples(df_candiate_triples, origin="MESH")

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
