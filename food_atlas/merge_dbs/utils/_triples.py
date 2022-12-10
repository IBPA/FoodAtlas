from copy import deepcopy

import pandas as pd


TAIL_METADATA = [
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
]


def _get_food_atlas_triples(row, triples, references):
    # Put common metadata first.
    triple = {
        'head': row['head'],
        'tail': row['tail'],
        'head_ncbi_taxid': row['head_ncbi_taxid'],
    }
    for tail_metadata in TAIL_METADATA:
        if f"tail_{tail_metadata}" in row.index:
            triple[f"tail_{tail_metadata}"] = row[f"tail_{tail_metadata}"]

    for reference_id in row['reference_ids']:
        try:
            reference_id = int(reference_id)
            reference = references.loc[reference_id]
        except KeyError:
            print(f"{reference_id} not found in references. Skipping.")
            continue
        except ValueError:
            print(f"{reference_id} is not a valid reference ID. Skipping.")
            continue

        triple_ = deepcopy(triple)
        triple_['reference_pmid'] = references.loc[reference_id]['pmid']
        triple_['reference_title'] \
            = references.loc[reference_id]['reference_name']
        triple_['confidence'] = 'high' \
            if references.loc[reference_id]['pmid'] != '' else 'low'

        triples += [triple_]


def get_food_atlas_triples(data, foods, chemicals, references):
    """
    """
    # Append food, chemical, and reference metadata to data content.
    data['head_ncbi_taxid'] = data['food_id'].apply(
        lambda fid: foods.loc[fid]['ncbi_taxid']
    )
    for tail_metadata in TAIL_METADATA:
        if tail_metadata in chemicals.columns:
            data[f'tail_{tail_metadata}'] = data['chemical_id'].apply(
                lambda cid: chemicals.loc[cid][tail_metadata]
            )

    triples = []
    data.apply(
        lambda row: _get_food_atlas_triples(row, triples, references),
        axis=1,
    )
    triples = pd.DataFrame(triples)

    return triples
