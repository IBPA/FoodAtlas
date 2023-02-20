import pandas as pd


def load_foods(validate=False):
    """Load the food data from the USDA Food Data Central database. Note that
    we are only considering Foundation Foods.

    """
    if validate:
        foundation_food = pd.read_csv("data/FDC/foundation_food.csv")
        fdc_ids_foundation_food = foundation_food['fdc_id'].unique()

        # FDC IDs that correspond to NCBI Taxonomy IDs.
        attributes = pd.read_csv("data/FDC/food_attribute.csv")
        attributes_foundation_food = attributes.query(
            "fdc_id in @fdc_ids_foundation_food"
        )
        attributes_foundation_food = pd.pivot_table(
            data=attributes_foundation_food,
            index='fdc_id',
            columns='name',
            values='value',
            aggfunc='first',
        )
        attributes_foundation_food['NDB_number'] \
            = foundation_food.set_index('fdc_id').loc[
                attributes_foundation_food.index]['NDB_number']
        attributes_foundation_food = attributes_foundation_food.reset_index()
        attributes_foundation_food.to_csv("foods_to_validate.csv")
        print(
            "Please validate the results in "
            "`data/FDC/foods_to_validate.csv`."
        )
    else:
        foods = pd.read_csv("data/FDC/foods_manually_validated.csv")
        foods = foods[foods['Skip'] != 1]

        def parse_ncbi_taxon(x):
            x = x.split('/')[-1]
            if x.startswith("NCBITaxon_"):
                return x[len("NCBITaxon_"):]
            elif x.startswith("NCBITAXON?p=classes&conceptid="):
                return x[len("NCBITAXON?p=classes&conceptid="):]
            else:
                try:
                    int(x)
                    return x
                except ValueError:
                    raise ValueError(f"Could not parse {x}")

        foods['ncbi_taxon'] = foods['NCBI Taxon'].apply(parse_ncbi_taxon)
        foods = foods.astype(
            {'fdc_id': 'Int64', 'ncbi_taxon': 'Int64'}
        ).set_index('fdc_id')

        return foods


def load_chemicals():
    chemicals = pd.read_csv(
        "data/FDC/chemicals_usda_unofficial.csv",
        encoding='latin-1',
    )
    chemicals = chemicals.rename(columns={
        'pubchem_compound_id': 'pubchem',
        'InChIKey_use': 'inchikey',
    })
    chemicals = chemicals.astype(
        {'id': 'Int64', 'pubchem': 'Int64'}
    ).set_index('id')

    return chemicals


def load_references():
    """So far we do not know how to map FDC IDs to FDC Sample IDs."""
    pass


def get_triples():
    """
    """
    foods = load_foods()
    chemicals = load_chemicals()

    data = pd.read_csv(
        "data/FDC/food_nutrient.csv",
        low_memory=False,
    )
    data = data.query(
        "fdc_id in @foods.index & nutrient_id in @chemicals.index"
    ).copy()
    data.to_csv("check_before.csv")
    data = data.query("amount > 0").copy()
    data['head'] = data['fdc_id'].apply(
        lambda x: foods.loc[x, 'FoodOn Ontology name for FDC item']
    )
    data['head_ncbi_taxid'] = data['fdc_id'].apply(
        lambda x: foods.loc[x, 'ncbi_taxon']
    )
    data['tail'] = data['nutrient_id'].apply(
        lambda x: chemicals.loc[x, 'NutrDesc']
    )
    data['tail_pubchem'] = data['nutrient_id'].apply(
        lambda x: chemicals.loc[x, 'pubchem']
    )
    data['tail_inchikey'] = data['nutrient_id'].apply(
        lambda x: chemicals.loc[x, 'inchikey']
    )

    data = data[[
        'head', 'tail', 'head_ncbi_taxid', 'tail_pubchem', 'tail_inchikey'
    ]].copy()
    data['confidence'] = 'low'

    return data


if __name__ == '__main__':
    triples = get_triples()
    triples.to_csv("data/FDC/fdc.tsv", sep='\t', index=False)

    print(triples['head'].unique().shape)
