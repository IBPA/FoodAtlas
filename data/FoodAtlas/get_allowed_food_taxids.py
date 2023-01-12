import sys
import pandas as pd


FOODB_FOOD_FILEPATH = "../FooDB/foodb_2020_04_07_csv/Food.csv"
FRIDA_FILEPATH = "../Frida/frida.tsv"
PE_FILEPATH = "../Phenol-Explorer/phenol_explorer.tsv"
ALLOWED_NCBI_TAXIDS_FILEPATH = "./allowed_ncbi_taxids.tsv"


def main():
    # FooDB
    df_foodb = pd.read_csv(FOODB_FOOD_FILEPATH, keep_default_na=False)
    foodb_taxids = df_foodb["ncbi_taxonomy_id"].tolist()

    # Frida
    df_frida = pd.read_csv(FRIDA_FILEPATH, sep='\t', keep_default_na=False)
    frida_taxids = df_frida["head_ncbi_taxid"].tolist()

    # Phenol-Explorer
    df_pe = pd.read_csv(PE_FILEPATH, sep='\t', keep_default_na=False)
    pe_taxids = df_pe["head_ncbi_taxid"].tolist()

    taxids = list(set(foodb_taxids + frida_taxids + pe_taxids))
    taxids.remove("")
    taxids = list(set([int(x) for x in taxids]))

    df = pd.DataFrame(taxids, columns=["ncbi_taxonomy_id"])
    df.to_csv(ALLOWED_NCBI_TAXIDS_FILEPATH, sep='\t', index=False)


if __name__ == "__main__":
    main()
