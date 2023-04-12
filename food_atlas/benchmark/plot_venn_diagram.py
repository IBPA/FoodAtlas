import numpy as np
import pandas as pd

from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt

from .utils import (
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


def map_sources_to_chemicals(data, evidence_annotated):
    data = data.copy()

    map_sources = evidence_annotated.groupby('sord_index').apply(
        lambda group: sorted(group['source'].unique().tolist())
        if group['label_p'].max() == 1
        else []
    )
    evidence_annotated['sources'] = evidence_annotated['sord_index'].apply(
        lambda x: map_sources.loc[x]
    )
    sources = evidence_annotated.groupby('mesh_id')['sources'].nth(0)
    mesh_to_sources = sources.to_dict()

    data['sources'] = data['mesh_id'].apply(
        lambda x: mesh_to_sources[x] if x in mesh_to_sources else np.nan
    )

    return data


def map_sources_to_inchikey_code(data):
    data = data.copy()

    map_code_to_sources = {}
    for inchikey_code in data['inchikey_structure_code'].unique():
        sources_all = []
        group = data.query(
            f"inchikey_structure_code == '{inchikey_code}'"
        )
        for sources in group['sources'].tolist():
            sources_all += sources
        sources_all = sorted(list(set(sources_all)))
        map_code_to_sources[inchikey_code] = sources_all

    data['sources_all'] = data['inchikey_structure_code'].apply(
        lambda x: map_code_to_sources[x]
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


def get_benchmark_results(data_fa, data_fm):
    pubchem_ids_fa = set(data_fa['pubchem_id'].tolist())
    pubchem_ids_fm = set(data_fm['pubchem_id'].tolist())

    result_rows = []
    for pubchem_id in pubchem_ids_fa.union(pubchem_ids_fm):
        label_p_se = data_fa.query(
            f"pubchem_id == {pubchem_id}"
        )['label_p']
        if len(label_p_se.value_counts()) > 1:
            raise ValueError(
                "There are multiple labels for the same chemical."
            )

        result_rows += [{
            'pubchem_id': pubchem_id,
            'in_foodmine': pubchem_id in pubchem_ids_fm,
            'in_foodatlas': pubchem_id in pubchem_ids_fa,
            'label_p': label_p_se.values[0] if len(label_p_se) > 0 else np.nan,
        }]
    result = pd.DataFrame(result_rows)

    return result


def plot_venn_cocoa():
    data_cocoa_fa, _, _, _ = load_cocoa_and_garlic_in_foodatlas()
    data_cocoa_fm, _ = load_cocoa_and_garlic_in_foodmine()

    # Load the entailed FoodAtlas chemicals.
    data_cocoa_annot = pd.read_excel(
        "outputs/benchmark/evidence_cocoa_annotated.xlsx",
        sheet_name='annotation',
    )
    data_cocoa_fa = map_labels_to_chemicals(data_cocoa_fa, data_cocoa_annot)
    data_cocoa_fa = data_cocoa_fa.query("label_p in [1, 0]")
    inchikeys_cocoa_fm = set(data_cocoa_fm['chem_id'].tolist())
    inchikeys_cocoa_fa = set(data_cocoa_fa['inchikey_structure_code'].tolist())

    venn2(
        subsets=(inchikeys_cocoa_fa, inchikeys_cocoa_fm),
        set_labels=('FoodAtlas', 'FoodMine'),
    )
    plt.savefig('outputs/benchmark/venn_cocoa_with_0.svg')
    plt.close()

    data_cocoa_fa = data_cocoa_fa.query("label_p in [1]")
    inchikeys_fm = set(data_cocoa_fm['chem_id'].tolist())
    inchikeys_fa = set(data_cocoa_fa['inchikey_structure_code'].tolist())
    venn2(
        subsets=(inchikeys_fa, inchikeys_fm),
        set_labels=('FoodAtlas', 'FoodMine'),
    )
    plt.savefig('outputs/benchmark/venn_cocoa.svg')
    plt.close()


def plot_venn_garlic():
    _, data_garlic_fa, _, _ = load_cocoa_and_garlic_in_foodatlas()
    _, data_garlic_fm = load_cocoa_and_garlic_in_foodmine()

    data_garlic_annot = pd.read_excel(
        "outputs/benchmark/evidence_garlic_annotated.xlsx",
        sheet_name='annotation',
    )
    data_garlic_fa = map_labels_to_chemicals(data_garlic_fa, data_garlic_annot)
    data_garlic_fa = data_garlic_fa.query("label_p in [1, 0]")
    inchikeys_garlic_fm = set(data_garlic_fm['chem_id'].tolist())
    inchikeys_garlic_fa = set(
        data_garlic_fa['inchikey_structure_code'].tolist()
    )

    venn2(
        subsets=(inchikeys_garlic_fa, inchikeys_garlic_fm),
        set_labels=('FoodAtlas', 'FoodMine'),
    )
    plt.savefig('outputs/benchmark/venn_garlic_with_0.svg')
    plt.close()

    data_garlic_fa = data_garlic_fa.query("label_p in [1]")
    inchikeys_fm = set(data_garlic_fm['chem_id'].tolist())
    inchikeys_fa = set(data_garlic_fa['inchikey_structure_code'].tolist())
    venn2(
        subsets=(inchikeys_fa, inchikeys_fm),
        set_labels=('FoodAtlas', 'FoodMine'),
    )
    plt.savefig('outputs/benchmark/venn_garlic.svg')
    plt.close()


def plot_venn3():
    data_cocoa_fa, data_garlic_fa, _, _ = load_cocoa_and_garlic_in_foodatlas()
    data_cocoa_fm, data_garlic_fm = load_cocoa_and_garlic_in_foodmine()

    for food in ['cocoa', 'garlic']:
        if food == 'cocoa':
            data_fa = data_cocoa_fa
            data_fm = data_cocoa_fm
        else:
            data_fa = data_garlic_fa
            data_fm = data_garlic_fm

        data_annot = pd.read_excel(
            f"outputs/benchmark/annotated/evidence_{food}_annotated.xlsx",
            sheet_name='annotation',
        )

        data_fa = map_labels_to_chemicals(data_fa, data_annot)
        data_fa = map_sources_to_chemicals(
            data_fa, data_annot).query("label_p in [1]")
        data_fa = map_sources_to_inchikey_code(data_fa)

        data_fa = data_fa.drop_duplicates(subset=['inchikey_structure_code'])

        chems_fm = set(data_fm['chem_id'].tolist())
        chems_fa = set()
        chems_exdb = set()

        def append_chems(row, chems_fa, chems_exdb):
            for source in row['sources_all']:
                if source.startswith('FoodAtlas'):
                    chems_fa.add(row['inchikey_structure_code'])
                else:
                    chems_exdb.add(row['inchikey_structure_code'])

        data_fa.apply(
            lambda row: append_chems(row, chems_fa, chems_exdb),
            axis=1,
        )

        # Also add the link prediction chemicals.
        chems_fa_lp = pd.read_excel(
            "outputs/benchmark/annotated/link_prediction_annotated.xlsx",
            sheet_name=food,
        ).query("Validation == 'Yes'")['inchikey_code'].tolist()
        chems_fa.update(chems_fa_lp)

        if food == 'cocoa':
            venn2(
                subsets=(chems_fa, chems_fm),
                set_labels=('FoodAtlas', 'FoodMine'),
            )
        else:
            venn3(
                subsets=(chems_fa, chems_fm, chems_exdb),
                set_labels=('FoodAtlas', 'FoodMine', 'External DBs'),
            )
        plt.savefig(f'outputs/benchmark/venn_{food}.svg')
        plt.close()


if __name__ == '__main__':
    plot_venn3()
