import numpy as np
import pandas as pd
from pandarallel import pandarallel

from ..data_processing.common_utils.knowledge_graph import KnowledgeGraph

pandarallel.initialize(progress_bar=True)


def _is_indexed(row, issns_indexed, journal_titles_indexed):
    if pd.notna(row['issn_formatted']) \
            and row['issn_formatted'] in issns_indexed:
        return True

    if pd.notna(row['journal_title_formatted']) \
            and row['journal_title_formatted'] in journal_titles_indexed:
        return True

    return np.nan


def is_indexed_in_agricola(references):
    indexed = pd.read_csv("food_atlas/benchmark/indexed_journals/agricola.csv")

    issns_indexed = indexed['ISSN (Print)'].str.strip().dropna().tolist() \
        + indexed['ISSN (online)'].str.strip().dropna().tolist()
    issns_indexed += ['2374-3832']
    issns_indexed = [x for x in issns_indexed if len(x) == 9]
    # Sanity check.
    for issn_indexed in issns_indexed:
        if len(issn_indexed) != 9:
            print(issn_indexed)

    journal_titles_indexed \
        = indexed['Title'].str.lower().str.strip().dropna().tolist()

    return references.apply(
        lambda row: _is_indexed(row, issns_indexed, journal_titles_indexed),
        axis=1,
    )


def is_indexed_in_cabi(references):
    data = pd.read_excel("food_atlas/benchmark/indexed_journals/cabi.xlsx")
    issns_indexed = data['ISSN'].str.strip().dropna().tolist()
    issns_indexed = [
        x if '(e-issn)' not in x else x[:9]
        for x in issns_indexed
    ]
    issns_indexed += ['1735-7179', '2322-1984', '2228-7795', '1682-2765']
    issns_indexed = [x for x in issns_indexed if len(x) == 9]
    # Sanity check.
    for issn_indexed in issns_indexed:
        if len(issn_indexed) != 9:
            print(issn_indexed)

    journal_titles_indexed \
        = data['Journal title '].str.lower().str.strip().dropna().tolist()

    return references.apply(
        lambda row: _is_indexed(row, issns_indexed, journal_titles_indexed),
        axis=1,
    )


def is_indexed_in_wos(references):
    data_dfs = []
    for cls in ['AHCI', 'ESCI', 'SCIE', 'SSCI']:
        data = pd.read_csv(
            f"food_atlas/benchmark/indexed_journals/wos_{cls}.csv",
        )
        data_dfs.append(data)
    data = pd.concat(data_dfs)
    issns_indexed \
        = data['ISSN'].str.strip().dropna().astype('str').tolist() \
        + data['eISSN'].str.strip().dropna().astype('str').tolist()
    # Sanity check.
    for issn_indexed in issns_indexed:
        if len(issn_indexed) != 9:
            print(issn_indexed)

    journal_titles_indexed \
        = data['Journal title'].str.lower().str.strip().dropna().tolist()

    return references.apply(
        lambda row: _is_indexed(row, issns_indexed, journal_titles_indexed),
        axis=1,
    )


def is_indexed_in_scopus(references):
    data = pd.read_csv("food_atlas/benchmark/indexed_journals/scopus.csv")
    issns_indexed \
        = data['Print-ISSN'].str.strip().dropna().astype('str').tolist() \
        + data['E-ISSN'].str.strip().dropna().astype('str').tolist()

    # Sanity check.
    for issn_indexed in issns_indexed:
        if len(issn_indexed) != 8:
            print(issn_indexed)

    journal_titles_indexed \
        = data[
            'Source Title (Medline-sourced journals are indicated in Green)'
        ].str.lower().str.strip().dropna().tolist()

    references = references.copy()
    references['issn_formatted'] = references['issn_formatted'].apply(
        lambda issn: issn.replace('-', '') if pd.notna(issn) else np.nan
    )

    return references.apply(
        lambda row: _is_indexed(row, issns_indexed, journal_titles_indexed),
        axis=1,
    )


def get_is_indexed(references):
    # Check AGRICOLA.
    references.loc[references['is_indexed'].isna(), 'is_indexed'] \
        = is_indexed_in_agricola(
            references.loc[references['is_indexed'].isna()]
        )

    # Check CABI.
    references.loc[references['is_indexed'].isna(), 'is_indexed'] \
        = is_indexed_in_cabi(
            references.loc[references['is_indexed'].isna()]
        )

    # Check WoS.
    references.loc[references['is_indexed'].isna(), 'is_indexed'] \
        = is_indexed_in_wos(
            references.loc[references['is_indexed'].isna()]
        )

    # Check Scopus.
    references.loc[references['is_indexed'].isna(), 'is_indexed'] \
        = is_indexed_in_scopus(
            references.loc[references['is_indexed'].isna()]
        )
    references['is_indexed'] = references['is_indexed'].fillna(False)

    return references


def load_references_frida():
    # If no ISSN and no PublicationName, then not indexed.
    references = pd.read_excel(
        'data/Frida/Frida_Dataset_June2022.xlsx', sheet_name='Source',
    )
    references['issn_formatted'] = references['ISSN'].str.strip()
    references['journal_title_formatted'] \
        = references['PublicationName'].str.lower().str.strip()
    references['is_indexed'] = np.nan

    # If PMID provided in ReportNumber, then indexed.
    references.loc[
        references['ReportNumber'].str.contains('PubMedID', na=False),
        'is_indexed'
    ] = True

    # No PMID, no ISSN, and no journal name, then not indexed.
    references.loc[
        (
            references['is_indexed'].isna()
            & references['ISSN'].isna()
            & references['PublicationName'].isna()
        ),
        'is_indexed'
    ] = False

    references = get_is_indexed(references)

    return references


def get_statistics_frida():
    references = load_references_frida()
    sources_indexed \
        = references.query("is_indexed == True")['SourceID'].tolist()

    print(f"# Ref.: {len(references)}")
    print(f"# Indexed Ref.: {len(sources_indexed)}")

    data = pd.read_excel(
        'data/Frida/Frida_Dataset_June2022.xlsx', sheet_name='Data_Normalised',
    )
    print(f"# Asso.: {len(data)}")

    data = data.query("Source.notnull() & Source != -1").copy()
    print(f"# Asso. with Ref.: {len(data)}")

    def is_indexed(source, sources_indexed):
        if type(source) == int:
            return source in sources_indexed
        elif type(source) == str:
            sources = source.split(', ')
            for source in sources:
                if int(source) in sources_indexed:
                    return True
            return False

    data['source_is_indexed'] = data['Source'].apply(
        lambda source: is_indexed(source, sources_indexed)
    )
    data = data.query("source_is_indexed == True").copy()
    print(f"# Asso. with Indexed Ref.: {len(data)}")

    foods = pd.read_excel(
        'data/Frida/Frida_Dataset_June2022.xlsx', sheet_name='Food',
    ).set_index('FoodID')
    data['food_is_indexed'] = data['FoodID'].apply(
        lambda x: foods.loc[x, ['FoodEx2Code', 'TaxonomicName']].notna().any()
    )
    # All checmicals are indexed with EuroFIR Code.
    data = data.query("food_is_indexed == True")
    print(
        f"# Asso. with Indexed Ref. and Indexed Food and Chemical: "
        f"{len(data)}"
    )


def load_references_phenol_explorer():
    references = pd.read_csv(
        "data/Phenol-Explorer/publications.csv", encoding='latin-1',
    )
    references['issn_formatted'] = np.nan
    references['journal_title_formatted'] \
        = references['journal_name'].str.lower().str.strip()
    # Phenol-Explorer uses literature from WoS according to the website.
    references['is_indexed'] = True

    # However, with manual inspection, the following 2 sources are not
    # from journals.
    references.loc[
        references['journal_title_formatted'].isin([
            'u.s. department of agriculture, agricultural research service',
            'unspecified',
        ]),
        'is_indexed'
    ] = False

    return references


def get_statistics_phenol_explorer():
    references = load_references_phenol_explorer()
    sources_indexed \
        = references.query("is_indexed == True")['id'].tolist()

    print(f"# Ref.: {len(references)}")
    print(f"# Indexed Ref.: {len(sources_indexed)}")

    data = pd.read_excel("data/Phenol-Explorer/composition-data.xlsx")
    print(f"# Asso.: {len(data)}")

    def is_indexed(source, sources_indexed):
        if type(source) == int:
            return source in sources_indexed
        elif type(source) == str:
            sources = source.split('; ')
            for source in sources:
                if int(source) in sources_indexed:
                    return True
            return False

    data['source_is_indexed'] = data['publication_ids'].apply(
        lambda source: is_indexed(source, sources_indexed)
    )
    data = data.query("source_is_indexed == True").copy()
    print(f"# Asso. with Indexed Ref.: {len(data)}")

    foods = pd.read_csv("data/Phenol-Explorer/foods.csv").set_index('name')
    chemicals = pd.read_csv(
        "data/Phenol-Explorer/compounds.csv"
    ).set_index('name')
    data['food_is_indexed'] = data['food'].apply(
        lambda x: pd.notna(foods.loc[x, 'food_source_scientific_name'])
    )
    data['chemical_is_indexed'] = data['compound'].apply(
        lambda x: chemicals.loc[
            x, ['cas_number', 'chebi_id', 'pubchem_compound_id']
        ].notna().any() if x in chemicals.index else False
    )
    data = data.query("food_is_indexed == True & chemical_is_indexed == True")
    print(
        f"# Asso. with Indexed Ref. and Indexed Food and Indexed Chemical: "
        f"{len(data)}"
    )
    print(data)


def get_statistics_fdc():
    data = pd.read_csv("data/FDC/food_nutrient.csv", low_memory=False)
    data = data.drop_duplicates(subset=['fdc_id', 'nutrient_id'])
    print(f"# Asso.: {len(data)}")


def get_statistics_foodatlas():
    kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
    evidence = kg.get_evidence()
    evidence = evidence.query(
        "relation == 'r0' "
        "& (source == 'FoodAtlas:annotation' "
        "| source == 'FoodAtlas:prediction:entailment')"
    ).copy()
    assert len(evidence.query("pmid == '' | pmid.isna()")) == 0
    print(f"Total number of PMIDs: {evidence['pmid'].nunique()}")

    evidence_annot = evidence.query(
        "source == 'FoodAtlas:annotation'").copy()
    print(f"Annotation - triplets: {evidence_annot['triple'].nunique()}")
    print(f"Annotation - pmids   : {evidence_annot['pmid'].nunique()}")

    evidence_entail = evidence.query(
        "source == 'FoodAtlas:prediction:entailment'").copy()
    print(f"Entailment - triplets: {evidence_entail['triple'].nunique()}")
    print(f"Entailment - pmids   : {evidence_entail['pmid'].nunique()}")


if __name__ == '__main__':
    get_statistics_frida()
    get_statistics_phenol_explorer()
    get_statistics_fdc()
    get_statistics_foodatlas()
