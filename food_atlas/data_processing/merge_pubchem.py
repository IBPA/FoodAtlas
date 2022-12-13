import argparse
import os
import requests
import sys
import warnings

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
CAS_ID_CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CID_QUERY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
CAS_CID_MAPPING_FILEPATH = "../../data/FoodAtlas/cas_cid_mapping.pkl"
CAS_INCHI_MAPPING_FILEPATH = "../../data/FoodAtlas/cas_inchi_mapping.pkl"
CAS_INCHIKEY_MAPPING_FILEPATH = "../../data/FoodAtlas/cas_inchikey_mapping.pkl"
CAS_CANONICAL_SMILES_MAPPING_FILEPATH = "../../data/FoodAtlas/cas_canonical_smiles_mapping.pkl"
CAS_MOLECULAR_FORMULA_MAPPING_FILEPATH = "../../data/FoodAtlas/cas_molecular_formula_mapping.pkl"


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
        cas_molecular_formula_mapping = load_pkl(CAS_MOLECULAR_FORMULA_MAPPING_FILEPATH)
    else:
        failed_cas_ids = []

        cas_cid_mapping = {}
        cas_inchi_mapping = {}
        cas_inchikey_mapping = {}
        cas_canonical_smiles_mapping = {}
        cas_molecular_formula_mapping = {}

        for cas_id in tqdm(cas_ids):
            url = CAS_ID_CID_QUERY_URL.format(cas_id)
            response = requests.get(url)
            if response.status_code != 200:
                failed_cas_ids.append(cas_id)
                warnings.warn(f"Error requesting data from {url}: {response.status_code}")
                continue

            response_json = response.json()
            cids = response_json["IdentifierList"]["CID"]
            cas_cid_mapping[cas_id] = cids

            for cid in cids:
                url = CID_QUERY_URL.format(cid)
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Error requesting data from {url}: {response.status_code}")

                response_json = response.json()
                record_section = response_json["Record"]["Section"]
                names_and_identifiers = _get_sections(record_section, "Names and Identifiers")
                computed_descriptors = _get_sections(names_and_identifiers, "Computed Descriptors")

                # InChI
                inchi = _get_values(computed_descriptors, "InChI")
                if cas_id in cas_inchi_mapping:
                    cas_inchi_mapping[cas_id].extend(inchi)
                else:
                    cas_inchi_mapping[cas_id] = inchi

                # InChIKey
                inchikey = _get_values(computed_descriptors, "InChIKey")
                if cas_id in cas_inchikey_mapping:
                    cas_inchikey_mapping[cas_id].extend(inchikey)
                else:
                    cas_inchikey_mapping[cas_id] = inchikey

                # Canonical SMILES
                canonical_smiles = _get_values(computed_descriptors, "Canonical SMILES")
                if cas_id in cas_canonical_smiles_mapping:
                    cas_canonical_smiles_mapping[cas_id].extend(canonical_smiles)
                else:
                    cas_canonical_smiles_mapping[cas_id] = canonical_smiles

                # molecular formula
                molecular_formula = _get_values(names_and_identifiers, "Molecular Formula")
                if cas_id in cas_molecular_formula_mapping:
                    cas_molecular_formula_mapping[cas_id].extend(molecular_formula)
                else:
                    cas_molecular_formula_mapping[cas_id] = molecular_formula

        print(f"Failed CAS IDs: {failed_cas_ids}")

        print(f"Saving pickled data to: {CAS_CID_MAPPING_FILEPATH}")
        save_pkl(cas_cid_mapping, CAS_CID_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_INCHI_MAPPING_FILEPATH}")
        save_pkl(cas_inchi_mapping, CAS_INCHI_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_INCHIKEY_MAPPING_FILEPATH}")
        save_pkl(cas_inchikey_mapping, CAS_INCHIKEY_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_CANONICAL_SMILES_MAPPING_FILEPATH}")
        save_pkl(cas_canonical_smiles_mapping, CAS_CANONICAL_SMILES_MAPPING_FILEPATH)

        print(f"Saving pickled data to: {CAS_MOLECULAR_FORMULA_MAPPING_FILEPATH}")
        save_pkl(cas_molecular_formula_mapping, CAS_MOLECULAR_FORMULA_MAPPING_FILEPATH)

    print(f"Found {len(cas_cid_mapping)}/{len(cas_ids)} CAS ID to cid mapping")
    print(f"Found {len(cas_inchi_mapping)}/{len(cas_ids)} CAS ID to inchi mapping")
    print(f"Found {len(cas_inchikey_mapping)}/{len(cas_ids)} CAS ID to inchikey mapping")
    print(f"Found {len(cas_canonical_smiles_mapping)}/{len(cas_ids)} CAS ID to canonical_smiles mapping")
    print(f"Found {len(cas_molecular_formula_mapping)}/{len(cas_ids)} CAS ID to molecular_formula mapping")

    print("Updating entities...")
    for cas_id in tqdm(cas_ids):
        cid = None
        inchi = None
        inchikey = None
        canonical_smiles = None
        molecular_formula = None

        if cas_id in cas_cid_mapping:
            cid = [str(x) for x in cas_cid_mapping[cas_id]]
        if cas_id in cas_inchi_mapping:
            inchi = [str(x) for x in cas_inchi_mapping[cas_id]]
        if cas_id in cas_inchikey_mapping:
            inchikey = [str(x) for x in cas_inchikey_mapping[cas_id]]
        if cas_id in cas_canonical_smiles_mapping:
            canonical_smiles = [str(x) for x in cas_canonical_smiles_mapping[cas_id]]
        if cas_id in cas_molecular_formula_mapping:
            molecular_formula = [str(x) for x in cas_molecular_formula_mapping[cas_id]]

        if [cid, inchi, inchikey, canonical_smiles, molecular_formula].count(None) == 5:
            continue

        entity = fa_kg.get_entity_by_other_db_id("CAS", cas_id)
        other_db_ids = entity["other_db_ids"]

        def _update_other_db_ids(key, val):
            if val is not None:
                if key in other_db_ids:
                    assert type(other_db_ids[key]) == list
                    other_db_ids[key].extend(val)
                else:
                    other_db_ids[key] = val

        _update_other_db_ids("PubChem", cid)
        _update_other_db_ids("InChI", inchi)
        _update_other_db_ids("InChIKey", inchikey)
        _update_other_db_ids("canonical_SMILES", canonical_smiles)
        _update_other_db_ids("molecular_formula", molecular_formula)

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
