import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph,  CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402
from common_utils.chemical_db_ids import get_pubchem_id_data_dict_using  # noqa: E402
from common_utils.chemical_db_ids import get_cas_cid_lookup_using_cas_ids  # noqa: E402

COMPATIBLE_FOOD_DBS = {"ncbi_taxid": "NCBI_taxonomy"}
COMPATIBLE_CHEMICAL_DBS = {"pubchem": "PubChem", "cas": "CAS"}
PUBCHEM_OTHER_DB_IDS_FILEPATH = "../../data/FoodAtlas/pubchem_other_db_ids.pkl"
CAS_OTHER_DB_IDS_FILEPATH = "../../data/FoodAtlas/cas_other_db_ids.pkl"


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
        "--external_db_filepath",
        type=str,
        required=True,
        help="Filepath of external DB.",
    )

    parser.add_argument(
        "--external_db_name",
        type=str,
        required=True,
        help="Name of external DB.",
    )

    parser.add_argument(
        "--new_pkl",
        action="store_true",
        help="Set if using new pickled data.",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        help="Number of workers for pandarallel.",
    )

    args = parser.parse_args()
    return args


def merge_other_db_ids(dict1, dict2):
    merged = deepcopy(dict1)

    for k, v in dict2.items():
        if type(v) is not list:
            raise RuntimeError()

        if k in merged:
            merged[k].extend(v)
        else:
            merged[k] = v

    return {k: sorted(list(set(v))) for k, v in merged.items()}


def main():
    args = parse_argument()

    if args.nb_workers is None:
        pandarallel.initialize(progress_bar=True)
    else:
        pandarallel.initialize(progress_bar=True, nb_workers=args.nb_workers)

    #
    df_to_add = pd.read_csv(args.external_db_filepath, sep='\t', keep_default_na=False)
    columns = list(df_to_add.columns)
    print(f"Columns: {columns}")

    data = []
    other_db_ids_dict = {x: [] for x in COMPATIBLE_CHEMICAL_DBS.values()}
    for _, row in df_to_add.iterrows():
        row_dict = {k: v for k, v in row.to_dict().items() if v != ''}

        compatible_food_columns = []
        for x in row_dict.keys():
            if x.startswith("head_") and x.replace("head_", "") in COMPATIBLE_FOOD_DBS.keys():
                compatible_food_columns.append(x)

        if len(compatible_food_columns) == 0:
            continue

        compatible_chemical_db_columns = []
        for x in row_dict.keys():
            if x.startswith("tail_") and x.replace("tail_", "") in COMPATIBLE_CHEMICAL_DBS.keys():
                compatible_chemical_db_columns.append(x)

        if len(compatible_chemical_db_columns) == 0:
            continue

        head_other_db_ids = \
            {v: [str(row_dict[f"head_{k}"])] for k, v in COMPATIBLE_FOOD_DBS.items()}
        head_other_db_ids["foodatlas_part_id"] = 'p0'

        head = CandidateEntity(
            type="organism",
            name=row_dict["head"],
            other_db_ids=head_other_db_ids,
        )

        relation = CandidateRelation(
            name='contains',
            translation='contains',
        )

        tail_other_db_ids = {
            v: [str(row_dict[f"tail_{k}"])]
            for k, v in COMPATIBLE_CHEMICAL_DBS.items()
            if f"tail_{k}" in row_dict
        }
        tail = CandidateEntity(
            type="chemical",
            name=row_dict["tail"],
            other_db_ids=tail_other_db_ids,
        )
        for k, v in tail_other_db_ids.items():
            other_db_ids_dict[k].extend(v)

        if "reference_pmid" in row_dict:
            pmid = row_dict["reference_pmid"]
            pmid = pmid.split('.')[0]
        else:
            pmid = ""

        if "reference_title" in row_dict:
            title = row_dict["reference_title"]
        else:
            title = ""

        data.append([head, relation, tail, pmid, title, row_dict["quality"]])

    df_triples = pd.DataFrame(
        data, columns=["head", "relation", "tail", "pmid", "title", "quality"])
    df_triples["source"] = args.external_db_name

    # add other DB ids
    other_db_ids_dict = {k: list(set(v)) for k, v in other_db_ids_dict.items()}
    pubchem_id_other_db_ids_dict = {}
    for db_name, db_ids in other_db_ids_dict.items():
        if db_name == "PubChem":
            new_pubchem_id_other_db_ids_dict = get_pubchem_id_data_dict_using(
                db_ids, what_data=['CAS', 'MESH'], using_type='PubChem')
        elif db_name == "CAS":
            cas_cid_lookup = get_cas_cid_lookup_using_cas_ids(db_ids)

            pubchem_ids = []
            for _, val in cas_cid_lookup.items():
                pubchem_ids.extend(val)
            pubchem_ids = list(set(pubchem_ids))
            new_pubchem_id_other_db_ids_dict = get_pubchem_id_data_dict_using(
                pubchem_ids, what_data=['CAS', 'MESH'], using_type='PubChem')
        else:
            raise ValueError()

        pubchem_id_other_db_ids_dict = {
            **pubchem_id_other_db_ids_dict, **new_pubchem_id_other_db_ids_dict}

    def _clean_other_db_ids(other_db_ids):
        return {
            k: v for k, v in other_db_ids.items()
            if k in ["MESH", "PubChem", "CAS"] and len(v) != 0
        }

    # now update triples
    def _f(row):
        head = row["head"]
        tail = row["tail"]
        newrows = []

        head_incompatible_dbs = set.intersection(
            set(COMPATIBLE_CHEMICAL_DBS.values()),
            set(head.other_db_ids)
        )
        if len(head_incompatible_dbs) != 0:
            raise NotImplementedError()

        pubchem_ids = []
        if "PubChem" in tail.other_db_ids:
            assert len(tail.other_db_ids["PubChem"]) == 1
            pubchem_id = tail.other_db_ids["PubChem"][0]
            if pubchem_id in pubchem_id_other_db_ids_dict:
                pubchem_ids.append(pubchem_id)

        if len(pubchem_ids) == 0 and "CAS" in tail.other_db_ids:
            assert len(tail.other_db_ids["CAS"]) == 1
            cas_id = tail.other_db_ids["CAS"][0]
            if cas_id in cas_cid_lookup:
                pubchem_ids.extend(cas_cid_lookup[cas_id])

        for pubchem_id in pubchem_ids:
            other_db_ids = pubchem_id_other_db_ids_dict[pubchem_id]
            other_db_ids['PubChem'] = [pubchem_id]
            other_db_ids = merge_other_db_ids(tail.other_db_ids, other_db_ids)
            row["tail"] = tail._replace(other_db_ids=_clean_other_db_ids(other_db_ids))
            newrows.append(deepcopy(row))

        return newrows

    #
    print("Updating chemical entities in the PH pairs...")
    results = []
    for result in df_triples.parallel_apply(_f, axis=1):
        results.extend(result)
    df_triples_updated = pd.DataFrame(results)

    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    fa_kg.add_triples(df_triples_updated)
    fa_kg.save(kg_dir=args.output_kg_dir)


if __name__ == "__main__":
    main()