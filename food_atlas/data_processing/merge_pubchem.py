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
QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
PUBCHEM_CID_MESH_FILEPATH = "../../data/PubChem/CID-MeSH.txt"
MESH_CAS_ID_PKL_FILEPATH = "../../data/FoodAtlas/mesh_cas_id_mapping.pkl"
MESH_INCHI_PKL_FILEPATH = "../../data/FoodAtlas/mesh_inchi_mapping.pkl"
MESH_INCHIKEY_PKL_FILEPATH = "../../data/FoodAtlas/mesh_inchikey_mapping.pkl"


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
        print("Using pickled data...")
        mesh_cas_id_mapping = load_pkl(MESH_CAS_ID_PKL_FILEPATH)
        mesh_inchi_mapping = load_pkl(MESH_INCHI_PKL_FILEPATH)
        mesh_inchikey_mapping = load_pkl(MESH_INCHIKEY_PKL_FILEPATH)
    else:
        with open(PUBCHEM_CID_MESH_FILEPATH, 'r') as _f:
            rows = _f.readlines()
        rows = [r.rstrip("\n").split('\t', 1) for r in rows]
        df_cid_mesh = pd.DataFrame(rows, columns=["CID", "MeSH_name"])

        mesh_cas_id_mapping = {}
        mesh_inchi_mapping = {}
        mesh_inchikey_mapping = {}
        for idx, row in tqdm(df_chemicals.iterrows(), total=df_chemicals.shape[0]):
            names_and_synonyms = [row["name"]] + row["synonyms"]
            names_and_synonyms = [x.lower() for x in names_and_synonyms]
            df_match = df_cid_mesh[df_cid_mesh["MeSH_name"].apply(
                lambda x: x.lower() in names_and_synonyms)]

            if df_match.shape[0] == 0:
                continue

            cas_ids = []
            inchi_list = []
            inchikey_list = []
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

                # CAS
                other_identifiers = \
                    [x for x in names_and_identifiers_section
                     if x["TOCHeading"] == "Other Identifiers"]
                if len(other_identifiers) == 1:
                    cas_section = other_identifiers[0]["Section"]
                    cas = [x for x in cas_section if x["TOCHeading"] == "CAS"]

                    if len(cas) == 1:
                        ids = [x["Value"]["StringWithMarkup"][0]["String"]
                               for x in cas[0]["Information"]]
                        cas_ids.append(list(set(ids)))

                # InChI & InChI key
                computed_descriptors = \
                    [x for x in names_and_identifiers_section
                     if x["TOCHeading"] == "Computed Descriptors"]
                if len(computed_descriptors) == 1:
                    computed_descriptors_section = computed_descriptors[0]["Section"]
                    inchi = [x for x in computed_descriptors_section if x["TOCHeading"] == "InChI"]

                    if len(inchi) == 1:
                        inchi = [x["Value"]["StringWithMarkup"][0]["String"]
                                 for x in inchi[0]["Information"]]
                        assert len(inchi) == 1
                        inchi_list.append(inchi[0])

                    inchikey = [x for x in computed_descriptors_section
                                if x["TOCHeading"] == "InChIKey"]
                    if len(inchikey) == 1:
                        inchikey = [x["Value"]["StringWithMarkup"][0]["String"]
                                    for x in inchikey[0]["Information"]]
                        assert len(inchikey) == 1
                        inchikey_list.append(inchikey[0])

            cas_ids = [x for sublist in cas_ids for x in sublist]
            cas_ids = list(set(cas_ids))
            inchi_list = list(set(inchi_list))
            inchikey_list = list(set(inchikey_list))

            mesh_id = row["other_db_ids"]["MESH"]
            if len(cas_ids) > 0:
                mesh_cas_id_mapping[mesh_id] = cas_ids
            if len(inchi_list) > 0:
                mesh_inchi_mapping[mesh_id] = inchi_list
            if len(inchikey_list) > 0:
                mesh_inchikey_mapping[mesh_id] = inchikey_list

        print(f"Saving pickled data to: {MESH_CAS_ID_PKL_FILEPATH}")
        save_pkl(mesh_cas_id_mapping, MESH_CAS_ID_PKL_FILEPATH)

        print(f"Saving pickled data to: {MESH_INCHI_PKL_FILEPATH}")
        save_pkl(mesh_inchi_mapping, MESH_INCHI_PKL_FILEPATH)

        print(f"Saving pickled data to: {MESH_INCHIKEY_PKL_FILEPATH}")
        save_pkl(mesh_inchikey_mapping, MESH_INCHIKEY_PKL_FILEPATH)

    print(f"Found {len(mesh_cas_id_mapping)}/{len(mesh_ids)} MeSH to CID mapping")
    print(f"Found {len(mesh_inchi_mapping)}/{len(mesh_ids)} MeSH to InChI mapping")
    print(f"Found {len(mesh_inchikey_mapping)}/{len(mesh_ids)} MeSH to InChIKey mapping")

    #
    assert len(set.intersection(*map(set, mesh_cas_id_mapping.values()))) == 0

    print("Updating entities...")
    for mesh_id in tqdm(mesh_ids):
        cas_id = None
        inchi = None
        inchikey = None
        if mesh_id in mesh_cas_id_mapping:
            cas_id = mesh_cas_id_mapping[mesh_id]
        if mesh_id in mesh_inchi_mapping:
            inchi = mesh_inchi_mapping[mesh_id]
        if mesh_id in mesh_inchikey_mapping:
            inchikey = mesh_inchikey_mapping[mesh_id]

        if [cas_id, inchi, inchikey].count(None) == 3:
            continue

        entity = fa_kg.get_entity_by_other_db_id("MESH", mesh_id)
        other_db_ids = entity["other_db_ids"]

        if cas_id:
            other_db_ids = {**other_db_ids, **{"CAS": cas_id}}
        if cas_id:
            other_db_ids = {**other_db_ids, **{"InChI": inchi}}
        if cas_id:
            other_db_ids = {**other_db_ids, **{"InChIKey": inchikey}}

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
