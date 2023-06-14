from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import requests
import os
import sys
from time import time
from typing import List
from urllib.request import urlopen
import urllib3
import warnings
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

from common_utils.utils import save_pkl, load_pkl

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FOODATLAS_DATA_DIR = "../../../data/FoodAtlas"
MESH_DATA_DIR = "../../../data/MESH"
DESC_FILEPATH = "../../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../../data/MESH/supp2022.xml"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"
PUBCHEM_CID_MESH_FILEPATH = "../../../data/PubChem/CID-MeSH.txt"
CAS_ID_CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
PUBCHEM_NAMES_FILEPATH = "../../../data/FoodAtlas/pubchem_names.pkl"
CAS_CID_LOOKUP_PKL_FILEPATH = "../../../data/FoodAtlas/cas_cid_lookup.pkl"
CID_JSON_LOOKUP_PKL_FILEPATH = "../../../data/FoodAtlas/cid_json_lookup.pkl"
NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH = "../../../data/FoodAtlas/ncbi_mesh_url_mesh_id_lookup.pkl"


def _get_cid_json_lookup_using_cid(cid: str):
    url = CID_QUERY_URL.format(cid)
    response = requests.get(url)
    retry_count = 0
    while response.status_code != 200:
        response = requests.get(url)
        retry_count += 1
        if retry_count == 10:
            raise RuntimeError(f"Error requesting data from {url}: {response.status_code}")

    return {cid: response.json()}


def get_cid_json_lookup_using_cids(cids: List[str], num_proc=4):
    assert len(cids) == len(set(cids))
    with Pool(processes=num_proc) as pool:
        r = list(tqdm(pool.imap(_get_cid_json_lookup_using_cid, cids), total=len(cids)))
    return {k: v for x in r for k, v in x.items()}


def get_ncbi_mesh_url_mesh_id_lookup_using_ncbi_mesh_url(ncbi_mesh_url: str):
    mesh_id = []
    try:
        soup = BeautifulSoup(urlopen(ncbi_mesh_url), features="html.parser")
    except Exception as e:
        print(f"Failed {ncbi_mesh_url}")
        return {}
    for paragraph in soup.find_all('p'):
        if paragraph.text.startswith("MeSH Unique ID: "):
            mesh_id.append(paragraph.text.split(": ")[1])

    if len(mesh_id) != 1:
        print(mesh_id)
        raise RuntimeError()
    return {ncbi_mesh_url: mesh_id[0]}


def get_ncbi_mesh_url_mesh_id_lookup_using_ncbi_mesh_urls(ncbi_mesh_urls: List[str], num_proc=3):
    assert len(ncbi_mesh_urls) == len(set(ncbi_mesh_urls))
    with Pool(processes=num_proc) as pool:
        r = list(tqdm(
            pool.imap(get_ncbi_mesh_url_mesh_id_lookup_using_ncbi_mesh_url, ncbi_mesh_urls),
            total=len(ncbi_mesh_urls)
        ))
    return {k: v for x in r for k, v in x.items()}


def get_pubchem_id_data_dict_using(
        using=List[str],
        what_data=List[str],
        using_type=str,
):
    if using_type.lower() == 'mesh':
        mesh_ids = using
        mesh_data_dict = read_mesh_data()
        mesh_names = []
        for x in mesh_ids:
            mesh_name = get_mesh_name_using_mesh_id(x, mesh_data_dict)
            if mesh_name:
                mesh_names.append(mesh_name)
        mesh_names = list(set(mesh_names))
        print(f"Found {len(mesh_names)} MeSH names from {len(mesh_ids)} MeSH IDs")

        df_cid_mesh = read_pubchem_cid_mesh()
        df_cid_mesh_match = df_cid_mesh[df_cid_mesh["mesh_name"].apply(lambda x: x in mesh_names)]
        pubchem_ids = list(set(df_cid_mesh_match["pubchem_id"].tolist()))
        print(f"Found {len(pubchem_ids)} PubChem IDs using {len(mesh_names)} MeSH names")
    elif using_type.lower() == 'pubchem':
        pubchem_ids = using
        print(f"Passed {len(pubchem_ids)} PubChem IDs")
    else:
        raise ValueError()

    if Path(CID_JSON_LOOKUP_PKL_FILEPATH).is_file():
        print(f"Loading pickled CID-JSON lookup file: {CID_JSON_LOOKUP_PKL_FILEPATH}")
        cid_json_lookup = load_pkl(CID_JSON_LOOKUP_PKL_FILEPATH)
    else:
        print(f"Initializing new CID-JSON lookup file: {CID_JSON_LOOKUP_PKL_FILEPATH}")
        cid_json_lookup = {}

    unpickled_pubchem_ids = [x for x in pubchem_ids if x not in cid_json_lookup]
    print(f"Length of previously unpickled PubChem IDs: {len(unpickled_pubchem_ids)}")
    if unpickled_pubchem_ids:
        new_cid_json_lookup = get_cid_json_lookup_using_cids(unpickled_pubchem_ids)
        cid_json_lookup = {**cid_json_lookup, **new_cid_json_lookup}
        save_pkl(cid_json_lookup, CID_JSON_LOOKUP_PKL_FILEPATH)

    # MeSH from NCBI URLs
    ncbi_mesh_urls = []
    for pubchem_id in pubchem_ids:
        ncbi_mesh_urls.extend(_get_ncbi_mesh_urls_from_json(cid_json_lookup[pubchem_id]))
    ncbi_mesh_urls = list(set(ncbi_mesh_urls))
    print(f"Found {len(ncbi_mesh_urls)} NCBI MeSH URLs")

    if Path(NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH).is_file():
        print(f"Loading pickled NCBI MeSH URL - MeSH ID lookup file: "
              f"{NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH}")
        ncbi_mesh_url_mesh_id_lookup = load_pkl(NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH)
    else:
        print(f"Initializing new NCBI MeSH URL - MeSH ID lookup file: "
              f"{NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH}")
        ncbi_mesh_url_mesh_id_lookup = {}

    unpickled_ncbi_mesh_urls = [x for x in ncbi_mesh_urls if x not in ncbi_mesh_url_mesh_id_lookup]
    print(f"Length of previously unpickled NCBI MeSH URLs: {len(unpickled_ncbi_mesh_urls)}")
    if unpickled_ncbi_mesh_urls:
        new_ncbi_mesh_url_mesh_id_lookup = get_ncbi_mesh_url_mesh_id_lookup_using_ncbi_mesh_urls(
            unpickled_ncbi_mesh_urls)
        ncbi_mesh_url_mesh_id_lookup = {
            **ncbi_mesh_url_mesh_id_lookup,
            **new_ncbi_mesh_url_mesh_id_lookup
        }
        save_pkl(ncbi_mesh_url_mesh_id_lookup, NCBI_MESH_URL_MESH_ID_LOOKUP_PKL_FILEPATH)

    pubchem_id_other_db_ids_dict = {}
    for pubchem_id in pubchem_ids:
        json = cid_json_lookup[pubchem_id]
        data = {}
        for d in what_data:
            if d == 'CAS':
                cas_ids = _get_cas_ids_from_json(json)
                if cas_ids:
                    data[d] = cas_ids
            elif d == 'MESH':
                ncbi_mesh_urls = _get_ncbi_mesh_urls_from_json(json)
                if ncbi_mesh_urls:
                    mesh_ids = [ncbi_mesh_url_mesh_id_lookup[x] for x in ncbi_mesh_urls]
                    data[d] = mesh_ids
            elif d == 'synonyms':
                synonyms = _get_synonyms_from_json(json)
                if synonyms:
                    data[d] = synonyms
            else:
                raise ValueError()
        pubchem_id_other_db_ids_dict[pubchem_id] = data.copy()

    return pubchem_id_other_db_ids_dict


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
        return [
            x["Value"]["StringWithMarkup"][0]["String"]
            for x in sections[0]["Information"]
        ]
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


def _get_name_from_json(json):
    return json["Record"]["RecordTitle"]


def _get_cas_ids_from_json(json):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    other_identifiers = _get_sections(names_and_identifiers, "Other Identifiers")
    if other_identifiers is not None:
        return list(set(_get_values(other_identifiers, "CAS")))
    else:
        return []


def _get_inchi_from_json(json):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")
    if computed_descriptors is None:
        raise RuntimeError()

    return list(set(_get_values(computed_descriptors, "InChI")))


def _get_inchikey_from_json(json):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")
    if computed_descriptors is None:
        raise RuntimeError()

    return list(set(_get_values(computed_descriptors, "InChIKey")))


def _get_canonical_smiles_from_json(json):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")
    if computed_descriptors is None:
        raise RuntimeError()

    return list(set(_get_values(computed_descriptors, "Canonical SMILES")))


def _get_ncbi_mesh_urls_from_json(json):
    record_section = json["Record"]["Section"]
    references = json["Record"]["Reference"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    synonyms = _get_sections(names_and_identifiers, "Synonyms")
    reference_numbers = []
    if synonyms is not None:
        reference_numbers = _get_reference_numbers(synonyms)

    ncbi_mesh_urls = []
    if len(reference_numbers) > 0:
        for x in references:
            if x["SourceName"] == "Medical Subject Headings (MeSH)" and \
               x["ReferenceNumber"] in reference_numbers:
                ncbi_mesh_urls.append(x["URL"])

    return ncbi_mesh_urls


def _get_synonyms_from_json(json, pubchem_top_n=None):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    synonyms = _get_sections(names_and_identifiers, "Synonyms")
    if synonyms is None:
        return []

    def _get_synonyms(tocheading):
        sections = [x for x in synonyms if x["TOCHeading"] == tocheading]
        if len(sections) == 1:
            values = [
                x["Value"]["StringWithMarkup"]
                for x in sections[0]["Information"]
            ]
            values = [x["String"] for x in values[0]]
            return values
        elif len(sections) == 0:
            return []
        else:
            raise RuntimeError()

    res = _get_synonyms("Depositor-Supplied Synonyms")
    if pubchem_top_n:
        res = res[:pubchem_top_n]

    res_lower = [x.lower() for x in res]
    mesh_entry_terms = _get_synonyms("MeSH Entry Terms")
    for x in mesh_entry_terms:
        if x.lower() not in res_lower:
            res.append(x)

    return res


def _get_summary_description_from_json(json):
    record_section = json["Record"]["Section"]

    names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
    if names_and_identifiers is None:
        raise RuntimeError()

    return _get_values(names_and_identifiers, "Record Description")


def make_pubchem_name_request(pubchem_id):
    result = query_using_cid(pubchem_id)
    if result is None:
        return (pubchem_id, None)

    return (pubchem_id, result["name"])


def _get_cas_cid_lookup_using_cas_id(cas_id: str):
    url = CAS_ID_CID_QUERY_URL.format(cas_id)
    response = requests.get(url)
    if response.status_code == 503:
        retry_count = 0
        while response.status_code == 503:
            response = requests.get(url)
            retry_count += 1
            if retry_count == 10:
                raise RuntimeError(f"Error requesting data from {url}: {response.status_code}")

    if response.status_code == 404:
        return {cas_id: []}
    elif response.status_code == 200:
        response_json = response.json()
        pubchem_ids = list(set([str(x) for x in response_json["IdentifierList"]["CID"]]))
        return {cas_id: sorted(pubchem_ids)}
    else:
        raise RuntimeError()


def get_cas_cid_lookup_using_cas_ids(cas_ids: List[str], num_proc=4):
    assert len(cas_ids) == len(set(cas_ids))

    if Path(CAS_CID_LOOKUP_PKL_FILEPATH).is_file():
        print(f"Loading cas_cid_lookup pickled file: {CAS_CID_LOOKUP_PKL_FILEPATH}")
        cas_cid_lookup = load_pkl(CAS_CID_LOOKUP_PKL_FILEPATH)
    else:
        print(f"Initializing new cas_cid_lookup: {CAS_CID_LOOKUP_PKL_FILEPATH}")
        cas_cid_lookup = {}

    print("Getting CAS-CID lookup using CAS IDs...")

    unpickled_cas_ids = [x for x in cas_ids if x not in cas_cid_lookup]
    print(f"Length of previously unpickled CAS IDs: {len(unpickled_cas_ids)}")
    if unpickled_cas_ids:
        with Pool(processes=num_proc) as pool:
            r = list(tqdm(
                pool.imap(_get_cas_cid_lookup_using_cas_id, cas_ids), total=len(cas_ids)
            ))
        new_cas_cid_lookup = {k: v for x in r for k, v in x.items()}
        cas_cid_lookup = {**cas_cid_lookup, **new_cas_cid_lookup}
        save_pkl(cas_cid_lookup, CAS_CID_LOOKUP_PKL_FILEPATH)

    result = {}
    for cas_id in cas_ids:
        result[cas_id] = cas_cid_lookup[cas_id]

    return result


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


def read_pubchem_cid_mesh():
    with open(PUBCHEM_CID_MESH_FILEPATH) as _f:
        cid_mesh = [line.rstrip() for line in _f]
    cid_mesh = [[x.split('\t')[0], x.split('\t')[1:]] for x in cid_mesh]
    df_cid_mesh = pd.DataFrame(cid_mesh, columns=["pubchem_id", "mesh_name"])
    df_cid_mesh = df_cid_mesh.explode("mesh_name", ignore_index=True)

    return df_cid_mesh


def read_mesh_data(
        mesh_data_dir=MESH_DATA_DIR,
        desc_filepath=DESC_FILEPATH,
        supp_filepath=SUPP_FILEPATH,
):
    # parse descriptors
    desc_mesh_id_tree_number_lookup_filepath = os.path.join(
        mesh_data_dir, 'desc_mesh_id_tree_number_lookup.pkl')
    desc_mesh_id_name_lookup_filepath = os.path.join(
        mesh_data_dir, 'desc_mesh_id_name_lookup.pkl')
    desc_tree_number_mesh_id_lookup_filepath = os.path.join(
        mesh_data_dir, 'desc_tree_number_mesh_id_lookup.pkl')
    desc_tree_number_name_lookup_filepath = os.path.join(
        mesh_data_dir, 'desc_tree_number_name_lookup.pkl')
    desc_mesh_id_entry_lookup_filepath = os.path.join(
        mesh_data_dir, 'desc_mesh_id_entry_lookup.pkl')

    if Path(desc_mesh_id_tree_number_lookup_filepath).is_file() and \
       Path(desc_mesh_id_name_lookup_filepath).is_file() and \
       Path(desc_tree_number_mesh_id_lookup_filepath).is_file() and \
       Path(desc_tree_number_name_lookup_filepath).is_file() and \
       Path(desc_mesh_id_entry_lookup_filepath).is_file():
        print("Found all pickle files for descriptor mesh data.")
        desc_mesh_id_tree_number_lookup = load_pkl(desc_mesh_id_tree_number_lookup_filepath)
        desc_mesh_id_name_lookup = load_pkl(desc_mesh_id_name_lookup_filepath)
        desc_tree_number_mesh_id_lookup = load_pkl(desc_tree_number_mesh_id_lookup_filepath)
        desc_tree_number_name_lookup = load_pkl(desc_tree_number_name_lookup_filepath)
        desc_mesh_id_entry_lookup = load_pkl(desc_mesh_id_entry_lookup_filepath)
    else:
        print("Loading descriptor XML...")
        desc = ET.parse(desc_filepath)
        desc_root = desc.getroot()

        print("Generating descriptor parent map...")
        desc_parent_map = {c: p for p in desc_root.iter() for c in p}

        print("Generating descriptor lookups...")
        desc_mesh_id_tree_number_lookup = {}
        desc_mesh_id_name_lookup = {}
        desc_mesh_id_entry_lookup = {}
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

            desc_mesh_id_entry_lookup[x.text] = []
            for y in parent.iter("String"):
                if desc_parent_map[y].tag != "Term":
                    continue
                desc_mesh_id_entry_lookup[x.text].append(y.text)

        desc_tree_number_mesh_id_lookup = {}
        for k, v in desc_mesh_id_tree_number_lookup.items():
            for x in v:
                desc_tree_number_mesh_id_lookup[x] = k

        desc_tree_number_name_lookup = {}
        for k, v in desc_tree_number_mesh_id_lookup.items():
            desc_tree_number_name_lookup[k] = desc_mesh_id_name_lookup[v]

        print('Saving desc pickles...')
        save_pkl(desc_mesh_id_tree_number_lookup, desc_mesh_id_tree_number_lookup_filepath)
        save_pkl(desc_mesh_id_name_lookup, desc_mesh_id_name_lookup_filepath)
        save_pkl(desc_tree_number_mesh_id_lookup, desc_tree_number_mesh_id_lookup_filepath)
        save_pkl(desc_tree_number_name_lookup, desc_tree_number_name_lookup_filepath)
        save_pkl(desc_mesh_id_entry_lookup, desc_mesh_id_entry_lookup_filepath)

    # parse supplementary
    supp_mesh_id_element_lookup_filepath = os.path.join(
        mesh_data_dir, 'supp_mesh_id_element_lookup.pkl')
    supp_mesh_id_heading_mesh_id_lookup_filepath = os.path.join(
        mesh_data_dir, 'supp_mesh_id_heading_mesh_id_lookup.pkl')
    supp_mesh_id_name_lookup_filepath = os.path.join(
        mesh_data_dir, 'supp_mesh_id_name_lookup.pkl')
    supp_mesh_id_entry_lookup_filepath = os.path.join(
        mesh_data_dir, 'supp_mesh_id_entry_lookup.pkl')

    if Path(supp_mesh_id_element_lookup_filepath).is_file() and \
       Path(supp_mesh_id_heading_mesh_id_lookup_filepath).is_file() and \
       Path(supp_mesh_id_name_lookup_filepath).is_file() and \
       Path(supp_mesh_id_entry_lookup_filepath).is_file():
        print("Found all pickle files for supplementary mesh data.")
        supp_mesh_id_element_lookup = load_pkl(supp_mesh_id_element_lookup_filepath)
        supp_mesh_id_heading_mesh_id_lookup = load_pkl(supp_mesh_id_heading_mesh_id_lookup_filepath)
        supp_mesh_id_name_lookup = load_pkl(supp_mesh_id_name_lookup_filepath)
        supp_mesh_id_entry_lookup = load_pkl(supp_mesh_id_entry_lookup_filepath)
    else:
        print("Loading supplementary XML...")
        supp = ET.parse(supp_filepath)
        supp_root = supp.getroot()

        print("Generating supplementary parent map...")
        supp_parent_map = {c: p for p in supp_root.iter() for c in p}

        print("Generating supplementary lookups...")
        supp_mesh_id_element_lookup = {}
        supp_mesh_id_heading_mesh_id_lookup = {}
        supp_mesh_id_name_lookup = {}
        supp_mesh_id_entry_lookup = {}
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

            supp_mesh_id_entry_lookup[x.text] = []
            for y in parent.iter("String"):
                if supp_parent_map[y].tag != "Term":
                    continue
                supp_mesh_id_entry_lookup[x.text].append(y.text)

        print('Saving supp pickles...')
        save_pkl(supp_mesh_id_element_lookup, supp_mesh_id_element_lookup_filepath)
        save_pkl(supp_mesh_id_heading_mesh_id_lookup, supp_mesh_id_heading_mesh_id_lookup_filepath)
        save_pkl(supp_mesh_id_name_lookup, supp_mesh_id_name_lookup_filepath)
        save_pkl(supp_mesh_id_entry_lookup, supp_mesh_id_entry_lookup_filepath)

    return {
        "desc_mesh_id_tree_number_lookup": desc_mesh_id_tree_number_lookup,
        "desc_mesh_id_name_lookup": desc_mesh_id_name_lookup,
        "desc_tree_number_mesh_id_lookup": desc_tree_number_mesh_id_lookup,
        "desc_tree_number_name_lookup": desc_tree_number_name_lookup,
        "desc_mesh_id_entry_lookup": desc_mesh_id_entry_lookup,
        "supp_mesh_id_element_lookup": supp_mesh_id_element_lookup,
        "supp_mesh_id_heading_mesh_id_lookup": supp_mesh_id_heading_mesh_id_lookup,
        "supp_mesh_id_name_lookup": supp_mesh_id_name_lookup,
        "supp_mesh_id_entry_lookup": supp_mesh_id_entry_lookup,
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
