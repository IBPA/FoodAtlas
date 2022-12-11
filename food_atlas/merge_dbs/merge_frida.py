import pandas as pd

from .utils import (
    get_ncbi_taxids,
    get_pmids,
    get_food_atlas_triples,
)


def load_foods(validate=False):
    """Load foods and process them.
    Steps:
    1. Remove TaxonomicName == 'NULL' and TaxonomicName == NaN.
    2. Remove foods and food groups that contain additives.
        - See `food_groups_ignored` below.
        - For the remaining, we extract only those with "raw" in the name.
    3. Enrich NCBI Taxonomy IDs.

    Args:
        validate (bool): If True, the function will return a DataFrame with
            the foods that need to be manually validated. If False, the
            function will return a DataFrame with the manually validated
            foods.

    Returns:
        foods (pd.DataFrame): A DataFrame with the foods.

    """
    if validate:
        foods = pd.read_excel(
            'data/Frida/Frida_Dataset_June2022.xlsx', sheet_name='Food',
        )
        foods = foods.query("TaxonomicName != 'NULL' & TaxonomicName.notna()")

        # Remove foods and food groups that contain additives.
        food_groups_ignored = [
            'Biscuits and cookies',
            'Boiled, smoked, cured or dried meat',
            'Breast milk and  infant formula',
            'Canned fruit products',
            'Canned legumes',
            'Canned vegetable products',
            'Cold cuts',
            'Condiments',
            'Fermented milk products',
            'Firm rennet cheese',
            'Marmelade, jelly etc.',
            'Other legume products',
            'Other meat and fresh meat products',
            'Other vegetable products',
            'Potato chip and snacks',
            'Processed cheese',
            'Unfermented milk products',
            'Yeast and baking powder',
        ]
        foods = foods.query('FoodGroup not in @food_groups_ignored')
        foods = foods.query('FoodName.str.contains("raw")')
        # Manually remove the only exception of the above.
        foods = foods.query('FoodName != "Strawberries, frozen, unsweetened"')
        foods = foods.copy()

        # Enrich NCBI Taxonomy IDs.
        sci_to_taxid = {}
        ncbi_taxids = get_ncbi_taxids(foods['TaxonomicName'].unique().tolist())
        for sci_name, taxid in zip(
                foods['TaxonomicName'].unique().tolist(),
                ncbi_taxids):
            sci_to_taxid[sci_name] = taxid
        foods['ncbi_taxid'] = foods['TaxonomicName'].map(sci_to_taxid)

        # Manually check the results.
        foods_to_validate = foods[
            ['FoodID', 'FoodName', 'TaxonomicName', 'ncbi_taxid']
        ]
        foods_to_validate = foods_to_validate.drop_duplicates('TaxonomicName')
        foods_to_validate.to_csv(
            "data/Frida/foods_to_validate.csv", index=False
        )
        print(
            "Please validate the results in "
            "`data/Frida/foods_to_validate.csv`."
        )
    else:
        foods = pd.read_csv(
            'data/Frida/foods_manually_validated.csv'
        )
        foods = foods.query("ncbi_taxid.notna()")
        foods = foods.astype(
            {'FoodID': 'Int64', 'ncbi_taxid': 'Int64'}
        ).set_index('FoodID')

    return foods


def load_chemicals():
    """Load chemicals.

    Returns:
        chemicals (pd.DataFrame): A DataFrame with the chemicals.

    """
    chemicals = pd.read_excel('data/Frida/ParameterDataDecember2022.xlsx')

    # Fix some errors in the data.
    chemicals.iloc[2664]['ParameterData'] = '7050-07-9'

    # Convert the data to a chemical look-up table.
    chemical_metadata_names = chemicals['ParameterDataType'].unique()
    chemical_dictrows = []
    for chemical_id in chemicals['ParameterID'].unique():
        if chemical_id == 'https://en.wikipedia.org/wiki/Iodine_in_biology':
            continue

        chemical = chemicals.query('ParameterID == @chemical_id')
        chemical_dictrow = {'ParameterID': chemical_id}
        for chemical_metadata_name in chemical_metadata_names:
            if pd.isna(chemical_metadata_name):
                continue

            try:
                chemical_metadata = chemical.query(
                    'ParameterDataType == @chemical_metadata_name'
                )
                if len(chemical_metadata) > 1:
                    chemical_metadata = ';'.join(chemical_metadata[
                        'ParameterData'].values)
                else:
                    chemical_metadata = chemical_metadata[
                        'ParameterData'].values[0]
                chemical_dictrow[chemical_metadata_name] = chemical_metadata
            except IndexError:
                chemical_dictrow[chemical_metadata_name] = None

        chemical_dictrows += [chemical_dictrow]

    chemicals = pd.DataFrame(chemical_dictrows).rename(
        columns={
            'PubChem_CID': 'pubchem',
            'CASNr': 'cas',
            'ChEBI': 'chebi',
            'ChEMBL': 'chembl',
            'KemiskSumformel': 'formula',
            'KemiskStruktur': 'smiles',
            'KEGG': 'kegg',
            'HMDB': 'hmdb',
            'EuroFIR Code': 'eurofir',
            'EFSA_PARAM_code': 'efsa_param',
            'ECHA_InfoCard': 'echa_infocard',
            'EC_Number': 'ec_number',  # Enzyme Commission Number
            'E_nummer': 'e_number',  # E(uropean) Number
        }
    ).astype({
        'ParameterID': 'Int64',
        'pubchem': 'Int64',
        'echa_infocard': 'Int64',
    }).set_index('ParameterID')
    chemicals = chemicals[[
        'pubchem',
        'cas',
        'chebi',
        'chembl',
        'formula',
        'smiles',
        'kegg',
        'hmdb',
        'eurofir',
        'efsa_param',
        'echa_infocard',
        'ec_number',
        'e_number',
    ]]

    return chemicals


def load_references():
    """Load references. Frida provides publications with different languages.
    For PMID enrichment, we only use English publications.

    """
    references = pd.read_excel(
        'data/Frida/Frida_Dataset_June2022.xlsx', sheet_name='Source',
    )

    # Enrich PMIDs. Check the available PMIDs first, then query the missing
    #   for English publications.
    def parse_pmid(x):
        if 'PubMedID' in str(x):
            return x.split('; ')[-1].split(': ')[-1]
        else:
            return ''
    references['pmid'] = references['ReportNumber'].apply(parse_pmid)
    references_eng = references.query(
        "Language == 'English' & pmid == ''"
    )
    titles_eng = references_eng['TitleEnglish'].unique().tolist()
    pmids_eng = get_pmids(titles_eng)

    title2pmid = {title: pmid for title, pmid in zip(titles_eng, pmids_eng)}
    references.loc[references_eng.index, 'pmid'] \
        = references_eng['TitleEnglish'].map(title2pmid)
    references['reference_name'] = references.apply(
        lambda row: f"{row['TitleEnglish']} ({row['Year']}) "
        f"[Publisher:{row['Publisher']},ISBN:{row['ISBN']},"
        f"ISSN:{row['ISSN']},PublicationName:{row['PublicationName']},"
        f"Volume:{row['Volume']},Issue:{row['Issue']},"
        f"PageStart:{row['PageStart']},PageEnd:{row['PageEnd']}]",
        axis=1,
    )
    references = references.set_index('SourceID')

    return references


def get_triples():
    foods = load_foods()
    chemicals = load_chemicals()
    references = load_references()

    data = pd.read_excel(
        'data/Frida/Frida_Dataset_June2022.xlsx',
        sheet_name='Data_Normalised',
    )
    print(f"Frida before cleaning: {len(data)} triples.")

    data = data.query("ResVal != 0")
    data = data.rename(columns={
        'FoodName': 'head',
        'ParameterName': 'tail',
        'FoodID': 'food_id',
        'ParameterID': 'chemical_id',
        'Source': 'reference_ids',
    })
    data['reference_ids'] = data['reference_ids'].apply(
        lambda x: [s.strip() for s in str(x).split(', ')]
    )
    data = data.query("food_id in @foods.index")
    data = data.query("chemical_id in @chemicals.index")
    print(f"Frida after cleaning: {len(data)} triples.")

    data = data.copy()
    triples = get_food_atlas_triples(data, foods, chemicals, references)
    triples.to_csv(
        'outputs/merge_dbs/food_chemical_triples/frida.tsv',
        sep='\t',
        index=False,
    )
    print(f"Frida adds {len(triples)} evidences.")
    print(triples['confidence'].value_counts())


if __name__ == '__main__':
    get_triples()
