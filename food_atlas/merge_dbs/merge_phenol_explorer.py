import logging
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    get_ncbi_taxids,
    get_pmids,
    get_food_atlas_triples,
)


logging.getLogger().setLevel(logging.INFO)
tqdm.pandas()


def load_foods(validate=False):
    if validate:
        foods = pd.read_csv("data/Phenol-Explorer/foods.csv")
        print(f"Phenol-Explorer - #foods - before cleaning: {len(foods)}")

        foods = foods.query("food_source_scientific_name.notna()")
        food_groups_ignored = [
            'Alcoholic beverages',
            'Coffee and cocoa',
        ]
        food_subgroups_ignored = [
            'Cereal products',
            'Cocoa beverage - Chocolate',
            'Coffee  beverage - Arabica Coffee  beverages',
            'Coffee  beverage - Robusta Coffee  beverages',
            'Coffee  beverage - Unknown Coffee  beverages',
            'Jams - Berry jams',
            'Jams - Drupe jams',
            'Jams - Pome jams',
            'Other seasonings',
            'Soy and soy products',
            'Soy drinks',
            'Spices - Spice blends',
            'Tea infusions',
        ]

        foods = foods.query(
            "food_group not in @food_groups_ignored & "
            "food_subgroup not in @food_subgroups_ignored",
        )
        print(f"Phenol-Explorer - #foods - after cleaning: {len(foods)}")

        foods = foods.copy()
        # Enrich NCBI Taxonomy IDs.
        sci_to_taxid = {}
        ncbi_taxids = get_ncbi_taxids(
            foods['food_source_scientific_name'].unique().tolist())
        for sci_name, taxid in zip(
                foods['food_source_scientific_name'].unique().tolist(),
                ncbi_taxids):
            sci_to_taxid[sci_name] = taxid
        foods['ncbi_taxid'] = foods['food_source_scientific_name'].map(
            sci_to_taxid)

        # Manually check the results.
        foods_to_validate = foods[
            ['id', 'name', 'food_source_scientific_name', 'ncbi_taxid']
        ]
        foods_to_validate = foods_to_validate.drop_duplicates(
            'food_source_scientific_name')
        foods_to_validate.to_csv(
            "data/Phenol-Explorer/foods_to_validate.csv", index=False
        )
        print(
            "Please validate the results in "
            "`data/Phenol-Explorer/foods_to_validate.csv`."
        )
    else:
        foods = pd.read_csv(
            'data/Phenol-Explorer/foods_manually_validated.csv'
        )
        foods = foods.query(
            "(ncbi_taxid != 'genus') & (ncbi_taxid != 'notfound') "
            "& (ncbi_taxid != 'invalid')"
        )
        foods = foods.astype(
            {'ncbi_taxid': 'Int64'}
        ).set_index('name').sort_index()

        return foods


def load_chemicals():
    chemicals = pd.read_csv("data/Phenol-Explorer/compounds.csv")

    chemicals = chemicals.rename(
        columns={
            'pubchem_compound_id': 'pubchem',
            'cas_number': 'cas',
            'chebi_id': 'chebi',
            'formula': 'formula',
        }
    ).astype({
        'pubchem': 'Int64',
        'chebi': 'Int64',
    })

    return chemicals.set_index('name').sort_index()


def load_references(validate=False):
    if validate:
        references = pd.read_csv(
            "data/Phenol-Explorer/publications.csv", encoding='latin-1'
        )

        # Ignore bad references.
        references = references.query("authors != 'UNSPECIFIED'").copy()

        # Enrich PMIDs.
        titles = references['title'].unique().tolist()
        pmids = get_pmids(titles)

        title2pmid = {title: pmid for title, pmid in zip(titles, pmids)}
        references.loc[references.index, 'pmid'] \
            = references['title'].map(title2pmid)
        references['reference_name'] = references.apply(
            lambda row: f"{row['authors']} "
            f"{row['title']} "
            f"({row['year_of_publication']}) "
            "["
            f"journal_name:{row['journal_name']},"
            f"journal_volume:{row['journal_volume']},"
            f"journal_issue:{row['journal_issue']},"
            f"pages:{row['pages']},"
            "]",
            axis=1,
        )

        # Manually check the results.
        references_to_validate = references[
            ['id', 'pmid', 'reference_name']
        ]
        references_to_validate.to_csv(
            "data/Phenol-Explorer/references_to_validate.csv", index=False
        )
        print(
            "Please validate the results in "
            "`data/Phenol-Explorer/references_to_validate.csv`."
        )
    else:
        references = pd.read_csv(
            'data/Phenol-Explorer/references_manually_validated.csv'
        )
        references['pmid'] = references['pmid'].apply(
            lambda x: str(int(x)) if not pd.isna(x) else ''
        )

        return references.set_index('id').sort_index()


def get_triples():
    """Transform the database content file into FoodAtlas triples.
    - Drop triples with concentration values of 0.

    """
    foods = load_foods()
    chemicals = load_chemicals()
    references = load_references()

    data = pd.read_excel("data/Phenol-Explorer/composition-data.xlsx")
    print(f"Phenol-Explorer before cleaning: {len(data)} triples.")

    # Did some sanity checks. We need to query PMIDs because the database does
    #   not match reference_id to PMID. Using our method, we have queried 651
    #   unique PMIDs, while the original database only provides 293. There are
    #   81 PMIDs in the original database that are not in our query results.
    #   We then did sanity check, and found that some of the 81 PMIDs are wrong
    #   (linking to different publications) and some are not found by our
    #   algorithm due to typos in the original database.
    # import numpy as np
    # ours = references['pmid'].dropna().astype(int).unique().tolist()
    # pudmed_ids = list(set([y.strip() for x in data['pubmed_ids'].tolist()
    #                       if not pd.isna(x) for y in str(x).split(';')]))
    # n_not_found = 0
    # for p in pudmed_ids:
    #     if int(p) not in ours:
    #         n_not_found += 1
    #         print(p)
    # print(len(pudmed_ids))
    # print(n_not_found)
    # print(len(ours))
    # print(len(set(pudmed_ids)))

    data = data.query("mean != 0").copy()
    data['food_id'] = data['food']
    data['chemical_id'] = data['compound']
    data = data.rename(columns={
        'food': 'head',
        'compound': 'tail',
        'publication_ids': 'reference_ids',
    })
    data['reference_ids'] = data['reference_ids'].apply(
        lambda x: [s.strip() for s in str(x).split('; ')]
    )
    data = data.query("food_id in @foods.index")
    data = data.query("chemical_id in @chemicals.index")
    print(f"Phenol-Explorer after cleaning: {len(data)} triples.")

    data = data.copy()
    triples = get_food_atlas_triples(data, foods, chemicals, references)
    triples.to_csv(
        'outputs/merge_dbs/food_chemical_triples/phenol_explorer.tsv',
        sep='\t',
        index=False,
    )
    print(f"Phenol-Explorer adds {len(triples)} evidences.")
    print(triples['confidence'].value_counts())


if __name__ == '__main__':
    triples = get_triples()
    # triples['database'] = 'Phenol-Explorer'
    # triples.to_csv(
    #     'outputs/merge_dbs/food_chemical_triples/phenol_explorer.tsv',
    #     sep='\t',
    #     index=False,
    # )
    # print(triples['confidence'].value_counts())
