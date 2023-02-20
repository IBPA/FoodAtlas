import csv
import sys
import pandas as pd
sys.path.append('../../food_atlas/data_processing')

from merge_ncbi_taxonomy import read_dmp_files  # noqa: E402
from common_utils.utils import load_pkl, save_pkl  # noqa: E402


FOODB_FOOD_FILEPATH = "../FooDB/foodb_2020_04_07_csv/Food.csv"
FRIDA_FILEPATH = "../Frida/frida.tsv"
PE_FILEPATH = "../Phenol-Explorer/phenol_explorer.tsv"
ALLOWED_NCBI_TAXIDS_FILEPATH = "./allowed_ncbi_taxids.tsv"
NCBI_NAMES_FILEPATH = "../NCBI_Taxonomy/names.dmp"
NCBI_NAME_CLASS_TO_USE = ["genbank common name", "scientific name", "common name"]
LITSENSE_QUERIES_FILEPATH = "./litsense_queries.txt"


def main():
    # FooDB
    df_foodb = pd.read_csv(FOODB_FOOD_FILEPATH, keep_default_na=False)
    df_foodb = df_foodb[df_foodb["ncbi_taxonomy_id"] != ""]
    foodb_taxids = df_foodb["ncbi_taxonomy_id"].tolist()

    foodb_foods = list(set(df_foodb["name"].tolist() + df_foodb["name_scientific"].tolist()))
    if "" in foodb_foods:
        foodb_foods.remove("")

    # Frida
    df_frida = pd.read_csv(FRIDA_FILEPATH, sep='\t', keep_default_na=False)
    frida_taxids = df_frida["head_ncbi_taxid"].tolist()

    # Phenol-Explorer
    df_pe = pd.read_csv(PE_FILEPATH, sep='\t', keep_default_na=False)
    pe_taxids = df_pe["head_ncbi_taxid"].tolist()

    taxids = list(set(foodb_taxids + frida_taxids + pe_taxids))
    taxids = list(set([int(x) for x in taxids]))
    df = pd.DataFrame(taxids, columns=["ncbi_taxonomy_id"])
    df.to_csv(ALLOWED_NCBI_TAXIDS_FILEPATH, sep='\t', index=False)

    # df_names = read_dmp_files(NCBI_NAMES_FILEPATH, filetype="names")
    # df_names = df_names[df_names["name_class"].apply(lambda x: x in NCBI_NAME_CLASS_TO_USE)]
    # df_names = df_names.groupby("tax_id")["name_txt"].apply(set).reset_index()

    df_names = load_pkl("/home/jasonyoun/Temp/df_names.pkl")
    df_names = df_names[df_names["tax_id"].apply(lambda x: int(x) in taxids)]
    queries = [f"\"{y}\" contains" for x in df_names["name_txt"].tolist() for y in x]
    queries += [f"\"{x}\" contains" for x in foodb_foods]
    queries = sorted(list(set([x.lower() for x in queries])))
    df_queries = pd.DataFrame(queries, columns=["query_string"])
    df_queries.to_csv(
        LITSENSE_QUERIES_FILEPATH, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    main()
