from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import requests
import os
import sys
from time import time
from urllib.request import urlopen
import urllib3
import warnings
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

from common_utils.utils import save_pkl, load_pkl

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FOODATLAS_DATA_DIR = "../../data/FoodAtlas"
MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"
PUBCHEM_CID_MESH_FILEPATH = "../../data/PubChem/CID-MeSH.txt"
CAS_ID_CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
PUBCHEM_OTHER_DB_IDS_FILEPATH = "../../data/FoodAtlas/pubchem_other_db_ids.pkl"
PUBCHEM_NAMES_FILEPATH = "../../data/FoodAtlas/pubchem_names.pkl"
CAS_OTHER_DB_IDS_FILEPATH = "../../data/FoodAtlas/cas_other_db_ids.pkl"
CAS_NAMES_FILEPATH = "../../data/FoodAtlas/cas_names.pkl"


def query_using_cid(cid: str):
    cas_id = []
    inchi = []
    inchikey = []
    canonical_smiles = []

    url = CID_QUERY_URL.format(cid)
    response = requests.get(url)
    retry_count = 0
    while response.status_code != 200:
        response = requests.get(url)
        retry_count += 1
        if retry_count == 10:
            raise RuntimeError(f"Error requesting data from {url}: {response.status_code}")
    response_json = response.json()
    record_section = response_json["Record"]["Section"]
    references = response_json["Record"]["Reference"]

    def _get_sections(input_list, tocheading):
        sections = [x for x in input_list if x["TOCHeading"] == tocheading]
        if len(sections) == 1:
            return sections[0]["Section"]
        elif len(sections) == 0:
            return None
        else:
            raise RuntimeError()

    def _get_values(input_list, tocheading):
        sections = [x for x in input_list if x["TOCHeading"] == tocheading]
        if len(sections) == 1:
            values = [
                x["Value"]["StringWithMarkup"][0]["String"]
                for x in sections[0]["Information"]
            ]
            return list(set(values))
        elif len(sections) == 0:
            return []
        else:
            raise RuntimeError()

    def _get_reference_numbers(input_list):
        sections = [x for x in input_list if x["TOCHeading"] == "MeSH Entry Terms"]
        if len(sections) == 1:
            rn = [x["ReferenceNumber"] for x in sections[0]["Information"]]
            return list(set(rn))
        elif len(sections) == 0:
            return []
        else:
            raise RuntimeError()

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")
    if computed_descriptors is None:
        raise RuntimeError()
    inchi = _get_values(computed_descriptors, "InChI")
    inchikey = _get_values(computed_descriptors, "InChIKey")
    canonical_smiles = _get_values(computed_descriptors, "Canonical SMILES")

    other_identifiers = _get_sections(names_and_identifiers, "Other Identifiers")
    if other_identifiers is not None:
        cas_id = _get_values(other_identifiers, "CAS")

    # mesh
    synonyms = _get_sections(names_and_identifiers, "Synonyms")
    reference_numbers = []
    if synonyms is not None:
        reference_numbers = _get_reference_numbers(synonyms)

    ncbi_urls = []
    if len(reference_numbers) > 0:
        for x in references:
            if x["SourceName"] == "Medical Subject Headings (MeSH)" and \
               x["ReferenceNumber"] in reference_numbers:
                ncbi_urls.append(x["URL"])

    mesh_id = []
    for ncbi_url in ncbi_urls:
        try:
            soup = BeautifulSoup(urlopen(ncbi_url), features="html.parser")
        except Exception as e:
            print(f"Failed for {cid}: {ncbi_url}")
            continue
        for paragraph in soup.find_all('p'):
            if paragraph.text.startswith("MeSH Unique ID: "):
                mesh_id.append(paragraph.text.split(": ")[1])
    mesh_id = list(set(mesh_id))

    return {
        "name": response_json["Record"]["RecordTitle"],
        "CAS": cas_id,
        "inchi": inchi,
        "inchikey": inchikey,
        "canonical_smiles": canonical_smiles,
        "mesh_id": mesh_id,
    }


def make_mesh_request(mesh_id, df_cid_mesh, supp_mesh_id_name_lookup, desc_mesh_id_name_lookup):
    # 1. get mesh name using MeSH DB
    if mesh_id.startswith('C'):
        if mesh_id not in supp_mesh_id_name_lookup:
            return (mesh_id, None)
        mesh_name = supp_mesh_id_name_lookup[mesh_id]
    elif mesh_id.startswith('D'):
        if mesh_id not in desc_mesh_id_name_lookup:
            return (mesh_id, None)
        mesh_name = desc_mesh_id_name_lookup[mesh_id]
    else:
        raise ValueError

    # 2. Use CID-MeSH to find all PubChem IDs
    pubchem_ids = df_cid_mesh[df_cid_mesh["mesh_name"] == mesh_name]
    pubchem_ids = list(set(pubchem_ids["pubchem_id"].tolist()))

    if len(pubchem_ids) == 0:
        return (mesh_id, None)

    cas_id_from_pubchem = []
    inchi_from_pubchem = []
    inchikey_from_pubchem = []
    canonical_smiles_from_pubchem = []
    mesh_id_from_pubchem = []
    # 3. For each PubChem ID use API to get MeSH Entry Terms
    # 4. Use the mesh URL link and get MeSH Unique ID
    for pubchem_id in pubchem_ids:
        result = query_using_cid(pubchem_id)
        if result is None:
            continue

        cas_id_from_pubchem.extend(result["CAS"])
        inchi_from_pubchem.extend(result["inchi"])
        inchikey_from_pubchem.extend(result["inchikey"])
        canonical_smiles_from_pubchem.extend(result["canonical_smiles"])
        mesh_id_from_pubchem.extend(result["mesh_id"])

    # 5. Match that MeSH Unique ID with original MeSH ID
    # 6. If match, add all the PubChemIDs
    if mesh_id not in mesh_id_from_pubchem:
        return (mesh_id, None)

    result = {
        "MESH": list(set(mesh_id_from_pubchem)),
        "PubChem": list(set(pubchem_ids)),
        "CAS": list(set(cas_id_from_pubchem)),
        "InChI": list(set(inchi_from_pubchem)),
        "InChIKey": list(set(inchikey_from_pubchem)),
        "canonical_SMILES": list(set(canonical_smiles_from_pubchem)),
    }

    return (mesh_id, result)


def make_pubchem_request(pubchem_id):
    result = query_using_cid(pubchem_id)
    if result is None:
        return (pubchem_id, None)

    result = {
        "MESH": list(set(result["mesh_id"])),
        "PubChem": [pubchem_id],
        "CAS": list(set(result["CAS"])),
        "InChI": list(set(result["inchi"])),
        "InChIKey": list(set(result["inchikey"])),
        "canonical_SMILES": list(set(result["canonical_smiles"])),
    }

    return (pubchem_id, result)


def make_pubchem_name_request(pubchem_id):
    result = query_using_cid(pubchem_id)
    if result is None:
        return (pubchem_id, None)

    return (pubchem_id, result["name"])


def make_cas_id_request(cas_id):
    cas_id_from_pubchem = []
    inchi_from_pubchem = []
    inchikey_from_pubchem = []
    canonical_smiles_from_pubchem = []
    mesh_id_from_pubchem = []

    url = CAS_ID_CID_QUERY_URL.format(cas_id)
    response = requests.get(url)
    if response.status_code != 200:
        warnings.warn(f"Error requesting data from {url}: {response.status_code}")
        return (cas_id, None)

    response_json = response.json()
    pubchem_ids = [str(x) for x in response_json["IdentifierList"]["CID"]]

    for pubchem_id in pubchem_ids:
        result = query_using_cid(pubchem_id)
        if result is None:
            continue

        cas_id_from_pubchem.extend(result["CAS"])
        inchi_from_pubchem.extend(result["inchi"])
        inchikey_from_pubchem.extend(result["inchikey"])
        canonical_smiles_from_pubchem.extend(result["canonical_smiles"])
        mesh_id_from_pubchem.extend(result["mesh_id"])

    if cas_id not in cas_id_from_pubchem:
        return (cas_id, None)

    result = {
        "MESH": list(set(mesh_id_from_pubchem)),
        "PubChem": list(set(pubchem_ids)),
        "CAS": list(set(cas_id_from_pubchem)),
        "InChI": list(set(inchi_from_pubchem)),
        "InChIKey": list(set(inchikey_from_pubchem)),
        "canonical_SMILES": list(set(canonical_smiles_from_pubchem)),
    }

    return (cas_id, result)


def make_cas_id_name_request(cas_id):
    cas_id_from_pubchem = []
    names = []

    url = CAS_ID_CID_QUERY_URL.format(cas_id)
    response = requests.get(url)
    if response.status_code != 200:
        warnings.warn(f"Error requesting data from {url}: {response.status_code}")
        return (cas_id, None)

    response_json = response.json()
    pubchem_ids = [str(x) for x in response_json["IdentifierList"]["CID"]]

    for pubchem_id in pubchem_ids:
        result = query_using_cid(pubchem_id)
        if result is None:
            continue

        cas_id_from_pubchem.extend(result["CAS"])
        names.append(result["name"])

    if cas_id not in cas_id_from_pubchem:
        return (cas_id, None)

    return (cas_id, list(set(names)))


def get_other_db_ids_using_cids(
        cids,
        new_pkl,
        pubchem_other_db_ids_filepath=PUBCHEM_OTHER_DB_IDS_FILEPATH,
        num_proc=None,
):
    if Path(pubchem_other_db_ids_filepath).is_file() and new_pkl is False:
        print(f"Loading pubchem_other_db_ids pickled file: {pubchem_other_db_ids_filepath}")
        pubchem_other_db_ids = load_pkl(pubchem_other_db_ids_filepath)
    else:
        print(f"Initializing new pubchem_other_db_ids: {pubchem_other_db_ids_filepath}")
        pubchem_other_db_ids = {}

    print("Getting other DB IDs using PubChem IDs...")

    original_shape = len(cids)
    cids = [x for x in cids if x not in pubchem_other_db_ids.keys()]
    print(f"{len(cids)}/{original_shape} are not in pickled file")

    with Pool(processes=cpu_count() if num_proc is None else num_proc) as pool:
        r = list(tqdm(pool.imap(make_pubchem_request, cids), total=len(cids)))
    other_db_ids = {x[0]: x[1] for x in r if x[1] is not None}

    for k, v in other_db_ids.items():
        if k not in pubchem_other_db_ids:
            pubchem_other_db_ids[k] = v

    save_pkl(pubchem_other_db_ids, pubchem_other_db_ids_filepath)

    return pubchem_other_db_ids


def get_pubchem_name_using_cids(
        cids,
        new_pkl,
        pubchem_names_filepath=PUBCHEM_NAMES_FILEPATH,
        num_proc=None,
):
    if Path(pubchem_names_filepath).is_file() and new_pkl is False:
        print(f"Loading pubchem_names pickled file: {pubchem_names_filepath}")
        pubchem_names = load_pkl(pubchem_names_filepath)
    else:
        print(f"Initializing new pubchem_names: {pubchem_names_filepath}")
        pubchem_names = {}

    print("Getting names using PubChem IDs...")

    original_shape = len(cids)
    cids = [x for x in cids if x not in pubchem_names.keys()]
    print(f"{len(cids)}/{original_shape} are not in pickled file")

    with Pool(processes=cpu_count() if num_proc is None else num_proc) as pool:
        r = list(tqdm(pool.imap(make_pubchem_name_request, cids), total=len(cids)))
    name = {x[0]: x[1] for x in r if x[1] is not None}

    for k, v in name.items():
        if k not in pubchem_names:
            pubchem_names[k] = v

    save_pkl(pubchem_names, pubchem_names_filepath)

    return pubchem_names


def get_other_db_ids_using_cas_ids(
        cas_ids,
        new_pkl,
        cas_other_db_ids_filepath=CAS_OTHER_DB_IDS_FILEPATH,
        num_proc=None,
):
    if Path(cas_other_db_ids_filepath).is_file() and new_pkl is False:
        print(f"Loading cas_other_db_ids pickled file: {cas_other_db_ids_filepath}")
        cas_other_db_ids = load_pkl(cas_other_db_ids_filepath)
    else:
        print(f"Initializing new cas_other_db_ids: {cas_other_db_ids_filepath}")
        cas_other_db_ids = {}

    print("Getting other DB IDs using CAS IDs...")

    original_shape = len(cas_ids)
    cas_ids = [x for x in cas_ids if x not in cas_other_db_ids.keys()]
    print(f"{len(cas_ids)}/{original_shape} are not in pickled file")

    with Pool(processes=cpu_count() if num_proc is None else num_proc) as pool:
        r = list(tqdm(pool.imap(make_cas_id_request, cas_ids), total=len(cas_ids)))
    other_db_ids = {x[0]: x[1] for x in r if x[1] is not None}

    for k, v in other_db_ids.items():
        if k not in cas_other_db_ids:
            cas_other_db_ids[k] = v

    save_pkl(cas_other_db_ids, cas_other_db_ids_filepath)

    return cas_other_db_ids


def get_pubchem_name_using_cas_ids(
        cas_ids,
        new_pkl,
        cas_names_filepath=CAS_NAMES_FILEPATH,
        num_proc=None,
):
    if Path(cas_names_filepath).is_file() and new_pkl is False:
        print(f"Loading cas_names pickled file: {cas_names_filepath}")
        cas_names = load_pkl(cas_names_filepath)
    else:
        print(f"Initializing new cas_names: {cas_names_filepath}")
        cas_names = {}

    print("Getting PubChem names using CAS IDs...")

    original_shape = len(cas_ids)
    cas_ids = [x for x in cas_ids if x not in cas_names.keys()]
    print(f"{len(cas_ids)}/{original_shape} are not in pickled file")

    with Pool(processes=cpu_count() if num_proc is None else num_proc) as pool:
        r = list(tqdm(pool.imap(make_cas_id_name_request, cas_ids), total=len(cas_ids)))
    other_db_ids = {x[0]: x[1] for x in r if x[1] is not None}

    for k, v in other_db_ids.items():
        if k not in cas_names:
            cas_names[k] = v

    save_pkl(cas_names, cas_names_filepath)

    return cas_names


def update_using_mesh(
        mesh_ids,
        df_cid_mesh,
        desc_mesh_id_name_lookup,
        supp_mesh_id_name_lookup,
):
    other_db_ids = {}
    with Pool(processes=cpu_count()) as pool:
        r = list(tqdm(pool.imap(partial(
                make_mesh_request, df_cid_mesh=df_cid_mesh,
                supp_mesh_id_name_lookup=supp_mesh_id_name_lookup,
                desc_mesh_id_name_lookup=desc_mesh_id_name_lookup
                ), mesh_ids),
            total=len(mesh_ids))
        )
    other_db_ids = {x[0]: x[1] for x in r if x[1] is not None}

    return other_db_ids


def read_pubchem_cid_mesh():
    with open(PUBCHEM_CID_MESH_FILEPATH) as _f:
        cid_mesh = [line.rstrip() for line in _f]
    cid_mesh = [[x.split('\t')[0], x.split('\t')[1:]] for x in cid_mesh]
    df_cid_mesh = pd.DataFrame(cid_mesh, columns=["pubchem_id", "mesh_name"])
    df_cid_mesh = df_cid_mesh.explode("mesh_name", ignore_index=True)

    return df_cid_mesh


def read_mesh_data(use_pkl):
    # parse descriptors
    desc_mesh_id_tree_number_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_tree_number_lookup.pkl')
    desc_mesh_id_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_mesh_id_name_lookup.pkl')
    desc_tree_number_mesh_id_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_tree_number_mesh_id_lookup.pkl')
    desc_tree_number_name_lookup_filepath = os.path.join(
        MESH_DATA_DIR, 'desc_tree_number_name_lookup.pkl')

    if use_pkl:
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
        desc_mesh_id_tree_number_lookup = {}
        desc_mesh_id_name_lookup = {}
        for x in desc_root.iter("DescriptorUI"):
            parent = desc_parent_map[x]
            if parent.tag != "DescriptorRecord":
                continue

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

    if use_pkl:
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

    return {
        "desc_mesh_id_tree_number_lookup": desc_mesh_id_tree_number_lookup,
        "desc_mesh_id_name_lookup": desc_mesh_id_name_lookup,
        "desc_tree_number_mesh_id_lookup": desc_tree_number_mesh_id_lookup,
        "desc_tree_number_name_lookup": desc_tree_number_name_lookup,
        "supp_mesh_id_element_lookup": supp_mesh_id_element_lookup,
        "supp_mesh_id_heading_mesh_id_lookup": supp_mesh_id_heading_mesh_id_lookup,
        "supp_mesh_id_name_lookup": supp_mesh_id_name_lookup,
    }


def get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict):
    desc_mesh_id_name_lookup = mesh_data_dict["desc_mesh_id_name_lookup"]
    supp_mesh_id_name_lookup = mesh_data_dict["supp_mesh_id_name_lookup"]

    if mesh_id.startswith('C'):
        if mesh_id not in supp_mesh_id_name_lookup:
            return None
        mesh_name = supp_mesh_id_name_lookup[mesh_id]
    elif mesh_id.startswith('D'):
        if mesh_id not in desc_mesh_id_name_lookup:
            return None
        mesh_name = desc_mesh_id_name_lookup[mesh_id]
    else:
        raise ValueError

    return mesh_name


def get_cid_using_mesh_id(mesh_id, df_cid_mesh, mesh_data_dict):
    mesh_name = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
    if mesh_name is None:
        return []

    df_cid_mesh_match = df_cid_mesh[df_cid_mesh["mesh_name"] == mesh_name]
    return df_cid_mesh_match["pubchem_id"].tolist()


def get_pubchem_id_other_db_ids_using_mesh(mesh_ids, new_pkl):
    mesh_data_dict = read_mesh_data(use_pkl=True)
    mesh_names = []
    for x in mesh_ids:
        mesh_name = get_mesh_name_using_mesh_id(x, mesh_data_dict)
        if mesh_name:
            mesh_names.append(mesh_name)
    mesh_names = list(set(mesh_names))
    print(f"Found {len(mesh_names)} MeSH names")

    df_cid_mesh = read_pubchem_cid_mesh()
    df_cid_mesh_match = df_cid_mesh[df_cid_mesh["mesh_name"].apply(lambda x: x in mesh_names)]
    pubchem_ids = list(set(df_cid_mesh_match["pubchem_id"].tolist()))
    pubchem_id_other_db_ids_dict = get_other_db_ids_using_cids(
        pubchem_ids, new_pkl=new_pkl, num_proc=4)

    return pubchem_id_other_db_ids_dict
