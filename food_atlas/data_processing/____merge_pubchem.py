import argparse
from copy import deepcopy
import os
import requests
import sys
from typing import List
import warnings
from urllib.request import urlopen

sys.path.append('..')

from bs4 import BeautifulSoup  # noqa: E402
from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph, CandidateEntity  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"
CAS_ID_CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
CAS_CID_MAPPING_FILEPATH = "../../data/PubChem/cas_cid_mapping.pkl"
CAS_INCHI_MAPPING_FILEPATH = "../../data/PubChem/cas_inchi_mapping.pkl"
CAS_INCHIKEY_MAPPING_FILEPATH = "../../data/PubChem/cas_inchikey_mapping.pkl"
CAS_CANONICAL_SMILES_MAPPING_FILEPATH = "../../data/PubChem/cas_canonical_smiles_mapping.pkl"


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


def _get_sections(input_list, tocheading):
    sections = [x for x in input_list if x["TOCHeading"] == tocheading]
    assert len(sections) == 1
    return sections[0]["Section"]


def _get_values(input_list, tocheading):
    sections = [x for x in input_list if x["TOCHeading"] == tocheading]
    assert len(sections) == 1
    values = [x["Value"]["StringWithMarkup"][0]["String"] for x in sections[0]["Information"]]
    return list(set(values))


def query_using_cid(cid: str):
    inchi = []
    inchikey = []
    canonical_smiles = []

    url = CID_QUERY_URL.format(cid)
    response = requests.get(url)

    try_count = 0
    while response.status_code != 200:
        response = requests.get(url)
        try_count += 1
        if try_count == 5:
            warnings.warn(f"Error requesting data from {url}: {response.status_code}")
            return None

    response_json = response.json()
    record_section = response_json["Record"]["Section"]
    references = response_json["Record"]["Reference"]
    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")

    inchi = _get_values(computed_descriptors, "InChI")
    inchikey = _get_values(computed_descriptors, "InChIKey")
    canonical_smiles = _get_values(computed_descriptors, "Canonical SMILES")

    ncbi_urls = []
    for x in references:
        if x["SourceName"] == "Medical Subject Headings (MeSH)" and x["SourceID"].isnumeric():
            ncbi_urls.append(x["URL"])

    mesh_id = []
    for ncbi_url in ncbi_urls:
        soup = BeautifulSoup(urlopen(ncbi_url), features="html.parser")
        for paragraph in soup.find_all('p'):
            if paragraph.text.startswith("MeSH Unique ID: "):
                mesh_id.append(paragraph.text.split(": ")[1])
    mesh_id = list(set(mesh_id))

    return {
        "inchi": inchi,
        "inchikey": inchikey,
        "canonical_smiles": canonical_smiles,
        "mesh_id": mesh_id,
    }


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

    cas_ids = []
    for x in df_chemicals["other_db_ids"].tolist():
        if "CAS" in x:
            cas_ids.extend([cas for cas in x["CAS"]])
    cas_ids = list(set(cas_ids))
    print(f"Number of unique CAS ids: {len(cas_ids)}")

    if args.use_pkl:
        print("Using pickled data...")
        cas_cid_mapping = load_pkl(CAS_CID_MAPPING_FILEPATH)
        cas_inchi_mapping = load_pkl(CAS_INCHI_MAPPING_FILEPATH)
        cas_inchikey_mapping = load_pkl(CAS_INCHIKEY_MAPPING_FILEPATH)
        cas_canonical_smiles_mapping = load_pkl(CAS_CANONICAL_SMILES_MAPPING_FILEPATH)
    else:
        cas_cid_mapping = {}
        cas_inchi_mapping = {}
        cas_inchikey_mapping = {}
        cas_canonical_smiles_mapping = {}

    pickled_cas_ids = list(cas_cid_mapping.keys())
    print(f"Number of known CAS IDs in pickled data: {len(pickled_cas_ids)}")

    for cas_id in tqdm(cas_ids):
        if cas_id in pickled_cas_ids:
            continue

        url = CAS_ID_CID_QUERY_URL.format(cas_id)
        response = requests.get(url)
        if response.status_code != 200:
            warnings.warn(f"Error requesting data from {url}: {response.status_code}")
            continue

        response_json = response.json()
        cids = response_json["IdentifierList"]["CID"]
        cas_cid_mapping[cas_id] = cids

        for cid in cids:
            inchi, inchikey, canonical_smiles = query_using_cid(cid)

            if cas_id in cas_inchi_mapping:
                cas_inchi_mapping[cas_id].extend(inchi)
            else:
                cas_inchi_mapping[cas_id] = inchi

            if cas_id in cas_inchikey_mapping:
                cas_inchikey_mapping[cas_id].extend(inchikey)
            else:
                cas_inchikey_mapping[cas_id] = inchikey

            if cas_id in cas_canonical_smiles_mapping:
                cas_canonical_smiles_mapping[cas_id].extend(canonical_smiles)
            else:
                cas_canonical_smiles_mapping[cas_id] = canonical_smiles

    if args.use_pkl:
        print(f"Saving pickled data to: {CAS_CID_MAPPING_FILEPATH}")
        save_pkl(cas_cid_mapping, CAS_CID_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_INCHI_MAPPING_FILEPATH}")
        save_pkl(cas_inchi_mapping, CAS_INCHI_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_INCHIKEY_MAPPING_FILEPATH}")
        save_pkl(cas_inchikey_mapping, CAS_INCHIKEY_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_CANONICAL_SMILES_MAPPING_FILEPATH}")
        save_pkl(cas_canonical_smiles_mapping, CAS_CANONICAL_SMILES_MAPPING_FILEPATH)

    print(f"Found {len(cas_cid_mapping)}/{len(cas_ids)} CAS ID to cid mapping")
    print(f"Found {len(cas_inchi_mapping)}/{len(cas_ids)} CAS ID to inchi mapping")
    print(f"Found {len(cas_inchikey_mapping)}/{len(cas_ids)} CAS ID to inchikey mapping")
    print(f"Found {len(cas_canonical_smiles_mapping)}/{len(cas_ids)} "
          f"CAS ID to canonical_smiles mapping")

    print("Updating entities...")
    entities_to_update = []
    for cas_id in tqdm(cas_ids):
        cid = None
        inchi = None
        inchikey = None
        canonical_smiles = None

        if cas_id in cas_cid_mapping:
            cid = [str(x) for x in cas_cid_mapping[cas_id]]
        if cas_id in cas_inchi_mapping:
            inchi = [str(x) for x in cas_inchi_mapping[cas_id]]
        if cas_id in cas_inchikey_mapping:
            inchikey = [str(x) for x in cas_inchikey_mapping[cas_id]]
        if cas_id in cas_canonical_smiles_mapping:
            canonical_smiles = [str(x) for x in cas_canonical_smiles_mapping[cas_id]]

        if [cid, inchi, inchikey, canonical_smiles].count(None) == 4:
            continue

        df_entity = fa_kg.get_entity_by_other_db_id("CAS", cas_id)
        assert df_entity.shape[0] == 1
        entity = df_entity.iloc[0]
        other_db_ids = entity["other_db_ids"]

        def _update_other_db_ids(key, val):
            if val is not None:
                if key in other_db_ids:
                    assert type(other_db_ids[key]) == list
                    other_db_ids[key].extend(val)
                    other_db_ids[key] = list(set(other_db_ids[key]))
                else:
                    other_db_ids[key] = val

        _update_other_db_ids("PubChem", list(set(cid)))
        _update_other_db_ids("InChI", list(set(inchi)))
        _update_other_db_ids("InChIKey", list(set(inchikey)))
        _update_other_db_ids("canonical_SMILES", list(set(canonical_smiles)))

        ent = CandidateEntity(
            type="chemical",
            other_db_ids=other_db_ids,
        )
        entities_to_update.append(ent)

    fa_kg.add_update_entities(entities_to_update)

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.output_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
