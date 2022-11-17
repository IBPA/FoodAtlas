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
QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
PUBCHEM_CID_MESH_FILEPATH = "../../data/PubChem/CID-MeSH.txt"
CTD_CHEMICALS_FILEPATH = "../../data/CTD/CTD_chemicals.csv"
MESH_CID_PKL_FILEPATH = "../../data/FoodAtlas/mesh_cas_id_mapping.pkl"
CTD_COLUMNS = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "Definition",
    "ParentIDs",
    "TreeNumbers",
    "ParentTreeNumbers",
    "Synonyms",
]


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

    if args.use_pkl:
        print(f"Using pickled data: {MESH_CID_PKL_FILEPATH}")
        mesh_cas_id_mapping = load_pkl(MESH_CID_PKL_FILEPATH)
    else:
        print("Generating data fresh.")
        # mesh to cid using CTD
        df_ctd_chemicals = pd.read_csv(
            CTD_CHEMICALS_FILEPATH,
            comment='#',
            names=CTD_COLUMNS,
            keep_default_na=False,
        )

        df_ctd_chemicals["ChemicalID"] = df_ctd_chemicals["ChemicalID"].apply(
            lambda x: x.lstrip("MESH:"))
        df_ctd_chemicals = df_ctd_chemicals[df_ctd_chemicals["CasRN"] != ""]
        mesh_cas_id_mapping = dict(zip(df_ctd_chemicals["ChemicalID"], df_ctd_chemicals["CasRN"]))
        mesh_cas_id_mapping = {k: [v] for k, v in mesh_cas_id_mapping.items() if k in mesh_ids}
        print(f"Found {len(mesh_cas_id_mapping)}/{len(mesh_ids)} MeSH to CID mapping using CTD")

        # mesh to cid using PubChem for the rest
        with open(PUBCHEM_CID_MESH_FILEPATH, 'r') as _f:
            rows = _f.readlines()
        rows = [r.rstrip("\n").split('\t', 1) for r in rows]
        df_cid_mesh = pd.DataFrame(rows, columns=["CID", "MeSH_name"])

        print("Using PubChem REST API to get CIDs")
        for idx, row in tqdm(df_chemicals.iterrows(), total=df_chemicals.shape[0]):
            mesh = row["other_db_ids"]["MESH"]
            if mesh in mesh_cas_id_mapping:
                continue

            df_match = df_cid_mesh[df_cid_mesh["MeSH_name"].apply(
                lambda x: x in [row["name"]] + row["synonyms"])]

            if df_match.shape[0] == 0:
                continue

            cas_ids = []
            matching_cids = list(set(df_match["CID"].tolist()))
            for cid in matching_cids:
                url = QUERY_URL.format(cid)
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Error requesting data from {url}: {response.status_code}")

                response_json = response.json()
                sections = response_json["Record"]["Section"]
                names_and_identifiers = \
                    [x for x in sections if x["TOCHeading"] == "Names and Identifiers"]
                assert len(names_and_identifiers) == 1
                names_and_identifiers_section = names_and_identifiers[0]["Section"]
                other_identifiers = \
                    [x for x in names_and_identifiers_section
                     if x["TOCHeading"] == "Other Identifiers"]
                if len(other_identifiers) == 0:
                    continue
                assert len(other_identifiers) == 1
                cas_section = other_identifiers[0]["Section"]
                cas = [x for x in cas_section if x["TOCHeading"] == "CAS"]
                if len(cas) == 0:
                    continue
                assert len(cas) == 1

                ids = [x["Value"]["StringWithMarkup"][0]["String"] for x in cas[0]["Information"]]
                cas_ids.append(list(set(ids)))

            cas_ids = [x for sublist in cas_ids for x in sublist]
            cas_ids = list(set(cas_ids))

            if len(cas_ids) == 0:
                continue

            mesh_cas_id_mapping[mesh] = cas_ids

        print(f"Saving pickled data to: {MESH_CID_PKL_FILEPATH}")
        save_pkl(mesh_cas_id_mapping, MESH_CID_PKL_FILEPATH)

    print(f"Found {len(mesh_cas_id_mapping)}/{len(mesh_ids)} "
          "MeSH to CID mapping using CTD and PubChem")

    #
    assert len(set.intersection(*map(set, mesh_cas_id_mapping.values()))) == 0

    for mesh, cas_ids in mesh_cas_id_mapping.items():
        entity = fa_kg.get_entity_by_other_db_id("MESH", mesh)
        other_db_ids = {**entity["other_db_ids"], **{"CAS": cas_ids}}
        fa_kg._update_entity(
            foodatlas_id=entity["foodatlas_id"],
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
