import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from matplotlib_venn import venn2
import matplotlib.pyplot as plt

from . import (
    load_cocoa_and_garlic_in_foodatlas,
    load_cocoa_and_garlic_in_foodmine,
)


def map_labels_to_chemicals(data, evidence_annotated):
    data = data.copy()

    labels = evidence_annotated.groupby('mesh_id')['label_p'].max()
    if labels.isnull().any():
        raise ValueError("There are missing labels in the data.")
    mesh_to_label = labels.to_dict()

    data['label_p'] = data['mesh_id'].apply(
        lambda x: mesh_to_label[x] if x in mesh_to_label else -2
    )

    return data


def map_foodmine_to_foodatlas_chemicals(data_fatlas, data_fmine):
    cids_fmine = set(data_fmine['pubchem_id'].tolist())
    inchikey_codes_fmine = set(data_fmine['chem_id'].tolist())

    def _is_in_fmine(row):
        if row['pubchem_id'] in cids_fmine:
            return 1
        elif row['inchikey_structure_code'] in inchikey_codes_fmine:
            return 1
        else:
            return 0

    data_fatlas = data_fatlas.copy()
    data_fatlas['found'] = data_fatlas.apply(
        _is_in_fmine,
        axis=1,
    )

    return data_fatlas


if __name__ == '__main__':
    data_cocoa_fa, data_garlic_fa, _, _ = load_cocoa_and_garlic_in_foodatlas()
    data_cocoa_fm, data_garlic_fm = load_cocoa_and_garlic_in_foodmine()

    # data_cocoa_fm.to_csv("foodmine_cocoa.csv")
    # data_garlic_fm.to_csv("foodmine_garlic.csv")

    # Cocoa.
    for food in ['cocoa', 'garlic']:
        if food == 'cocoa':
            data_fa = data_cocoa_fa
            data_fm = data_cocoa_fm
        else:
            data_fa = data_garlic_fa
            data_fm = data_garlic_fm

        data_annot = pd.read_excel(
            f"outputs/benchmark/evidence_{food}_annotated.xlsx",
            sheet_name='annotation',
        )
        data_fa = map_labels_to_chemicals(data_fa, data_annot)
        data_fa = data_fa.query("label_p in [1]")

        chem_ids_fa = set(data_fa['inchikey_structure_code'].tolist())
        chem_ids_fm_excl = set(data_fm['chem_id']) - chem_ids_fa
        chem_ids_fm_common = set(data_fm['chem_id']) & chem_ids_fa

        data_fm_excl = data_fm.query(
            f"chem_id in {list(chem_ids_fm_excl)}"
        )
        data_fm_common = data_fm.query(
            f"chem_id in {list(chem_ids_fm_common)}"
        )

        pmids_excl = []
        for x in data_fm_excl['papers'].tolist():
            pmids_excl += x
        pmids_excl_set = set(pmids_excl)

        pmids_common = []
        for x in data_fm_common['papers'].tolist():
            pmids_common += x
        pmids_common_set = set(pmids_common)

        print(f"Food: {food}")
        print(pmids_common_set)
        print(len(pmids_common_set))
        print(len(pmids_excl_set))
        print(len(pmids_excl_set & pmids_common_set))

        # data_cocoa_fm_ex_papers_rows = []
        # for pubmed_id in pubmed_ids:
        #     row = {
        #         'pubmed_id': pubmed_id,
        #         'comment': '',
        #     }
        #     data_cocoa_fm_ex_papers_rows += [row]
        # data_cocoa_fm_ex_papers = pd.DataFrame(data_cocoa_fm_ex_papers_rows)
        # data_cocoa_fm_ex_papers.to_csv("foodmine_cocoa_ex_papers.csv")

    # # data_cocoa_fm_ex.to_csv("foodmine_cocoa_ex.csv")
