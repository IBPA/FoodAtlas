import os

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

Entrez.email = "fzli@ucdavis.edu"


STOP_WORDS = ['(', ')']
VARIANT_TERMS = [
    'sp.', 'ssp.', 'subsp.', 'var.',
    'sp', 'ssp', 'subsp', 'var',
]
CROSS_BREED_TERMS = ['x', u'\xd7']
# These terms are usually sensitive to the context, thus require manual checks.
INVALID_TERMS = [
    'convar.',
    'convar',
]


def format_query(scientific_name):
    """Format the scientific name to query the NCBI taxonomy database."""
    # Remove stop words.
    for stop_word in STOP_WORDS:
        scientific_name = scientific_name.replace(stop_word, '')

    terms = [term for term in scientific_name.split(' ') if term]

    # Skip cross breeds.
    for cb_term in CROSS_BREED_TERMS:
        if f" {cb_term} " in scientific_name:
            return "CROSS_BREED"

    # Several rules to filter out invalid scientific names.
    # 1. Impossible to form genus and species with less than 2 terms.
    if len(terms) < 2:
        return "INVALID_SCIENTIFIC_NAME"

    # 2. The first 2 terms must not be abbreviations.
    if '.' in terms[0] or '.' in terms[1]:
        return "INVALID_SCIENTIFIC_NAME"

    # 3. If invalid terms exist, then manually checking is required.
    for term in terms:
        if term in INVALID_TERMS:
            return "INVALID_SCIENTIFIC_NAME"

    # 4. If there are no abbreviations, such as authority terms or variant
    #   terms, then the scientific name is unlikely to be longer than 2 terms.
    if len(terms) > 2 and '.' not in scientific_name:
        return "INVALID_SCIENTIFIC_NAME"

    # 5. If dot exists in a term, then it must be in the end.
    for term in terms:
        if '.' in term and term[-1] != '.':
            return "INVALID_SCIENTIFIC_NAME"

    if len(terms) == 2:
        return f"{terms[0]} {terms[1]}"
    else:
        # Remove authority terms.
        query_terms = []
        for i, term in enumerate(terms[2:]):
            if term in VARIANT_TERMS:
                # Add dot if not exists.
                if term[-1] != '.':
                    term += '.'
                # Rename to "subsp." for subspecies.
                if term in ['sp.', 'ssp.']:
                    term = 'subsp.'
                # Add the immediate next term.
                query_terms += [term, terms[2 + i + 1]]
                break

        return f"{terms[0]} {terms[1]} {' '.join(query_terms)}"


def query_ncbi_taxid(scientific_name):
    with Entrez.esearch(db='taxonomy', term=scientific_name) as handle:
        record = Entrez.read(handle)

    return record['IdList']


def get_ncbi_taxids(scientific_names):
    if not os.path.exists("data/_CACHED_QUERIES/_sci_name_to_ncbi_taxid.csv"):
        hash_table = pd.DataFrame()
    else:
        hash_table = pd.read_csv(
            "data/_CACHED_QUERIES/_sci_name_to_ncbi_taxid.csv",
            index_col='food',
        ).astype({'ncbi_taxid': 'Int64'})

    ncbi_taxids = []
    hash_entries_new = []
    for food in tqdm(scientific_names):
        food = food.lower()
        food_query = format_query(food)

        # If the food name is already in the hash table, retrieve it.
        try:
            ncbi_taxids += [hash_table.loc[food]['ncbi_taxid']]
            continue
        except KeyError:
            pass

        ncbi_taxid = query_ncbi_taxid(food_query)
        if len(ncbi_taxid) == 1:
            ncbi_taxid = ncbi_taxid[0]
        elif len(ncbi_taxid) > 1:
            print(f"WARNING: {food} has multiple NCBI taxids: {ncbi_taxid}.")
            ncbi_taxid = ';'.join(ncbi_taxid)
        else:
            ncbi_taxid = ''

        ncbi_taxids += [ncbi_taxid]
        hash_entries_new += [{
            'food': food,
            'query': food_query,
            'ncbi_taxid': ncbi_taxid
        }]

    if hash_entries_new:
        hash_table = pd.concat(
            [hash_table, pd.DataFrame(hash_entries_new).set_index('food')],
        ).sort_index()
        hash_table = hash_table.reset_index().drop_duplicates(
            'food', keep='first'
        ).set_index('food')
        hash_table.to_csv(
            "data/_CACHED_QUERIES/_sci_name_to_ncbi_taxid.csv"
        )

    return ncbi_taxids
