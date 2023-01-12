import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from merge_ncbi_taxonomy import read_dmp_files
from common_utils.utils import load_pkl, save_pkl
from common_utils.chemical_db_ids import get_pubchem_name_using_cids
from common_utils.chemical_db_ids import get_pubchem_name_using_cas_ids

NCBI_ID_NAMES_DICT_PKL_FILEPATH = "../../data/NCBI_Taxonomy/ncbi_id_names_dict.pkl"
NCBI_NAMES_FILEPATH = "../../data/NCBI_Taxonomy/names.dmp"
NCBI_NAME_CLASS_TO_USE = ["genbank common name", "scientific name", "common name"]


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Filepath of the input data (either frida.tsv or phenol_explorer.tsv)."
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        required=True,
        help="Filepath of the output data (either frida_queries.tsv or phenol_explorer_queries.tsv)."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    #
    df = pd.read_csv(args.input_filepath, sep="\t")
    df_queries = df[["head", "head_ncbi_taxid", "tail", "tail_pubchem", "tail_cas"]].copy()
    df_queries["head_ncbi_taxid"] = df_queries["head_ncbi_taxid"].astype(str)
    df_queries["tail_pubchem"] = df_queries["tail_pubchem"].apply(
        lambda x: np.nan if np.isnan(x) else str(int(x))
    )
    df_queries = df_queries.dropna(subset=["tail_pubchem", "tail_cas"], thresh=1)

    # food
    print("Reading names.dmp...")
    if Path(NCBI_ID_NAMES_DICT_PKL_FILEPATH).is_file():
        print(f"Loading pickled file: {NCBI_ID_NAMES_DICT_PKL_FILEPATH}")
        ncbi_id_names_dict = load_pkl(NCBI_ID_NAMES_DICT_PKL_FILEPATH)
    else:
        df_names = read_dmp_files(NCBI_NAMES_FILEPATH, filetype="names")
        df_names = df_names[df_names["name_class"].apply(lambda x: x in NCBI_NAME_CLASS_TO_USE)]
        df_names = df_names.groupby("tax_id")["name_txt"].apply(set).reset_index()
        ncbi_id_names_dict = dict(zip(df_names["tax_id"].tolist(), df_names["name_txt"].tolist()))
        print(f"Saving pickle file: {NCBI_ID_NAMES_DICT_PKL_FILEPATH}")
        save_pkl(ncbi_id_names_dict, NCBI_ID_NAMES_DICT_PKL_FILEPATH)

    def _get_head_name(row):
        names = list(ncbi_id_names_dict[row["head_ncbi_taxid"]])
        names = [x.lower().strip() for x in names]
        return names

    df_queries["head_name"] = df_queries.apply(lambda row: _get_head_name(row), axis=1)

    # chem
    pubchem_ids = list(set(df_queries["tail_pubchem"].tolist()))
    pubchem_ids.remove(np.nan)
    pubchem_id_name_dict = get_pubchem_name_using_cids(pubchem_ids, new_pkl=False, num_proc=4)
    pubchem_id_name_dict = {k: [v] for k, v in pubchem_id_name_dict.items()}

    cas_ids = list(set(df_queries[df_queries["tail_pubchem"].isna()]["tail_cas"].tolist()))
    assert np.nan not in cas_ids
    cas_id_name_dict = get_pubchem_name_using_cas_ids(cas_ids, new_pkl=False, num_proc=4)

    def _get_tail_name(row):
        if type(row["tail_pubchem"]) == float:
            names = list(set(cas_id_name_dict[row["tail_cas"]] + [row["tail"]]))
        else:
            names = list(set(pubchem_id_name_dict[row["tail_pubchem"]] + [row["tail"]]))
        return [x.lower() for x in names]

    df_queries["tail_name"] = df_queries.apply(lambda row: _get_tail_name(row), axis=1)

    # pairs
    df_queries = df_queries.explode('head_name')
    df_queries = df_queries.explode('tail_name')

    def _make_query_string(row):
        query_str = f"{row['head_name']} {row['tail_name']}"
        return query_str.strip()

    df_queries["query_string"] = df_queries.apply(lambda row: _make_query_string(row), axis=1)
    df_queries = df_queries[["query_string"]].drop_duplicates()
    df_queries.to_csv(args.output_filepath, sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
