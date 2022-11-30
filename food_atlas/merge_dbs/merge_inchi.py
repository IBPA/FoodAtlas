import argparse
import os
import requests
import sys

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
INCHI_QUERY_URL = "https://cactus.nci.nih.gov/chemical/structure/{}/stdinchi"
INCHIKEY_QUERY_URL = "https://cactus.nci.nih.gov/chemical/structure/{}/stdinchikey"


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

    print("Updating entities...")
    for _, row in tqdm(df_chemicals.iterrows(), total=df_chemicals.shape[0]):
        other_db_ids = row["other_db_ids"]
        if "CAS" not in other_db_ids:
            continue

        assert "InChI" not in other_db_ids
        assert "InChIKey" not in other_db_ids

        inchi_list = []
        inchikey_list = []
        for cas_id in other_db_ids["CAS"]:
            # inchi
            url = INCHI_QUERY_URL.format(cas_id)
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"Error requesting data from {url}: {response.status_code}")

            inchi = response.text
            assert inchi.startswith("InChI=")
            inchi_list.append(inchi)

            # inchikey
            url = INCHIKEY_QUERY_URL.format(cas_id)
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"Error requesting data from {url}: {response.status_code}")

            inchikey = response.text
            assert inchikey.startswith("InChIKey=")
            inchikey_list.append(inchikey)

        other_db_ids = {
            **other_db_ids,
            **{"InChI": inchi_list},
            **{"InChIKey": inchikey_list},
        }

        fa_kg._update_entity(
            foodatlas_id=row["foodatlas_id"],
            other_db_ids=other_db_ids,
        )

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
