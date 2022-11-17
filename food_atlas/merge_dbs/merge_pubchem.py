import argparse
import itertools
import os
import requests
import sys
import xml.etree.ElementTree as ET

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.knowledge_graph import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
PUBCHEM_CID_MESH_FILEPATH = "../../data/PubChem/CID-MeSH.txt"
QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

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

    count = 0

    # mesh to cid
    with open(PUBCHEM_CID_MESH_FILEPATH, 'r') as _f:
        rows = _f.readlines()
    rows = [r.rstrip("\n").split('\t', 1) for r in rows]
    df_cid_mesh = pd.DataFrame(rows, columns=["CID", "MeSH_name"])

    mesh_cid_map = {}
    for idx, row in df_chemicals.iterrows():
        mesh_name_synonyms = [row["name"]] + row["synonyms"]

        df_match = df_cid_mesh[df_cid_mesh["MeSH_name"].apply(lambda x: x in mesh_name_synonyms)]

        if df_match.shape[0] != 1:
            print("Match is not shape 1.")
            print(row)
            print(df_match)
            sys.exit()

        matching_cid = df_match.iloc[0]["CID"]
        url = QUERY_URL.format(matching_cid)
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Error requesting data from {url}: {response.status_code}")

        response_json = response.json()
        references = response_json["Record"]["Reference"]
        cas_ids = [x["SourceID"] for x in references if x["SourceName"] == "CAS Common Chemistry"]

        if len(cas_ids) != 1:
            print("Length of cas_ids is not 1.")
            print(cas_ids)
            print(row)
            print(df_match)
            print()
            count += 1

        # assert cas_id.count('-') == 2

        # mesh_cid_map[row["other_db_ids"]["MESH"]] = cas_id

    print("Count: ", count)

    # print(mesh_cid_map)


if __name__ == '__main__':
    main()
