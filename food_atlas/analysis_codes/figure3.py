from collections import Counter
from itertools import product
import math
import os
from pathlib import Path
import sys
import textwrap

sys.path.append('../data_processing/')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from tqdm import tqdm  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from upsetplot import UpSet  # noqa: E402

FINAL_DATA_DIR = "../../outputs/backend_data/v0.1"
OUTPUT_DIR = "../../outputs/analysis_codes/figure3"
BACKEND_DATA_DIR = "../../outputs/backend_data/v0.1"


def visualize_source(fa_kg):
    df_evidence = fa_kg.get_evidence()
    print(df_evidence)
    sources = list(set(df_evidence['source'].tolist()))
    print(f"Sources: {sources}")

    df_fa = df_evidence[df_evidence["source"].apply(lambda x: x.startswith("FoodAtlas"))]
    print(f"Number of df_fa: {df_fa.shape[0]}")
    df_frida = df_evidence[df_evidence["source"].apply(lambda x: x == "Frida")]
    print(f"Number of df_frida: {df_frida.shape[0]}")
    df_phenol_explorer = df_evidence[df_evidence["source"].apply(lambda x: x == "Phenol-Explorer")]
    print(f"Number of df_phenol_explorer: {df_phenol_explorer.shape[0]}")
    df_fdc = df_evidence[df_evidence["source"].apply(lambda x: x == "FDC")]
    print(f"Number of df_fdc: {df_fdc.shape[0]}")
    df_mesh = df_evidence[df_evidence["source"].apply(lambda x: x == "MeSH")]
    print(f"Number of df_mesh: {df_mesh.shape[0]}")
    df_ncbi = df_evidence[df_evidence["source"].apply(lambda x: x == "NCBI_taxonomy")]
    print(f"Number of df_ncbi: {df_ncbi.shape[0]}")

    # high
    print()
    df_fa_high = df_fa[df_fa["quality"] == "high"]
    print(f"Number of df_fa_high: {df_fa_high.shape[0]}")
    df_frida_high = df_frida[df_frida["quality"] == "high"]
    print(f"Number of df_frida_high: {df_frida_high.shape[0]}")
    df_phenol_explorer_high = df_phenol_explorer[df_phenol_explorer["quality"] == "high"]
    print(f"Number of df_phenol_explorer_high: {df_phenol_explorer_high.shape[0]}")
    df_fdc_high = df_fdc[df_fdc["quality"] == "high"]
    print(f"Number of df_fdc_high: {df_fdc_high.shape[0]}")
    df_mesh_high = df_mesh[df_mesh["quality"] == "high"]
    print(f"Number of df_mesh_high: {df_mesh_high.shape[0]}")
    df_ncbi_high = df_ncbi[df_ncbi["quality"] == "high"]
    print(f"Number of df_ncbi_high: {df_ncbi_high.shape[0]}")

    # medium
    print()
    df_fa_medium = df_fa[df_fa["quality"] == "medium"]
    print(f"Number of df_fa_medium: {df_fa_medium.shape[0]}")
    df_frida_medium = df_frida[df_frida["quality"] == "medium"]
    print(f"Number of df_frida_medium: {df_frida_medium.shape[0]}")
    df_phenol_explorer_medium = df_phenol_explorer[df_phenol_explorer["quality"] == "medium"]
    print(f"Number of df_phenol_explorer_medium: {df_phenol_explorer_medium.shape[0]}")
    df_fdc_medium = df_fdc[df_fdc["quality"] == "medium"]
    print(f"Number of df_fdc_medium: {df_fdc_medium.shape[0]}")
    df_mesh_medium = df_mesh[df_mesh["quality"] == "medium"]
    print(f"Number of df_mesh_medium: {df_mesh_medium.shape[0]}")
    df_ncbi_medium = df_ncbi[df_ncbi["quality"] == "medium"]
    print(f"Number of df_ncbi_medium: {df_ncbi_medium.shape[0]}")

    # low
    print()
    df_fa_low = df_fa[df_fa["quality"] == "low"]
    print(f"Number of df_fa_low: {df_fa_low.shape[0]}")
    df_frida_low = df_frida[df_frida["quality"] == "low"]
    print(f"Number of df_frida_low: {df_frida_low.shape[0]}")
    df_phenol_explorer_low = df_phenol_explorer[df_phenol_explorer["quality"] == "low"]
    print(f"Number of df_phenol_explorer_low: {df_phenol_explorer_low.shape[0]}")
    df_fdc_low = df_fdc[df_fdc["quality"] == "low"]
    print(f"Number of df_fdc_low: {df_fdc_low.shape[0]}")
    df_mesh_low = df_mesh[df_mesh["quality"] == "low"]
    print(f"Number of df_mesh_low: {df_mesh_low.shape[0]}")
    df_ncbi_low = df_ncbi[df_ncbi["quality"] == "low"]
    print(f"Number of df_ncbi_low: {df_ncbi_low.shape[0]}")

    df_fa_high.drop_duplicates("triple", inplace=True)
    df_fa_medium.drop_duplicates("triple", inplace=True)
    df_fa_low.drop_duplicates("triple", inplace=True)
    df_frida_high.drop_duplicates("triple", inplace=True)
    df_frida_medium.drop_duplicates("triple", inplace=True)
    df_frida_low.drop_duplicates("triple", inplace=True)
    df_phenol_explorer_high.drop_duplicates("triple", inplace=True)
    df_phenol_explorer_medium.drop_duplicates("triple", inplace=True)
    df_phenol_explorer_low.drop_duplicates("triple", inplace=True)
    df_fdc_high.drop_duplicates("triple", inplace=True)
    df_fdc_medium.drop_duplicates("triple", inplace=True)
    df_fdc_low.drop_duplicates("triple", inplace=True)
    df_mesh_high.drop_duplicates("triple", inplace=True)
    df_mesh_medium.drop_duplicates("triple", inplace=True)
    df_mesh_low.drop_duplicates("triple", inplace=True)
    df_ncbi_high.drop_duplicates("triple", inplace=True)
    df_ncbi_medium.drop_duplicates("triple", inplace=True)
    df_ncbi_low.drop_duplicates("triple", inplace=True)

    data = [
        ["FoodAtlas - High", df_fa_high.shape[0]],
        ["FoodAtlas - Medium", df_fa_medium.shape[0]],
        ["FoodAtlas - Low", df_fa_low.shape[0]],
        ["Frida - High", df_frida_high.shape[0]],
        ["Frida - Medium", df_frida_medium.shape[0]],
        ["Frida - Low", df_frida_low.shape[0]],
        ["Phenol-Explorer - High", df_phenol_explorer_high.shape[0]],
        ["Phenol-Explorer - Medium", df_phenol_explorer_medium.shape[0]],
        ["Phenol-Explorer - Low", df_phenol_explorer_low.shape[0]],
        ["FDC - High", df_fdc_high.shape[0]],
        ["FDC - Medium", df_fdc_medium.shape[0]],
        ["FDC - Low", df_fdc_low.shape[0]],
        ["MeSH - High", df_mesh_high.shape[0]],
        ["MeSH - Medium", df_mesh_medium.shape[0]],
        ["MeSH - Low", df_mesh_low.shape[0]],
        ["NCBI - High", df_ncbi_high.shape[0]],
        ["NCBI - Medium", df_ncbi_medium.shape[0]],
        ["NCBI - Low", df_ncbi_low.shape[0]],
    ]

    df = pd.DataFrame(data, columns=["Source-Quality", "Count"])
    print(df)

    fig = px.bar(
        df,
        x="Source-Quality",
        y="Count",
        log_y=True,
        text="Count",
        width=500,
        height=400,
    )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "quality_sources.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "quality_sources.svg"))


def upset_plot(fa_kg):
    df_evidence = fa_kg.get_evidence()

    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:prediction:entailment', 'FoodAtlas:pred:ent')
    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:prediction:lp', 'FoodAtlas:pred:LP')
    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:annotation', 'FoodAtlas:annot')

    sources = sorted(list(set(df_evidence['source'].tolist())))
    print(sources)

    df_relation = fa_kg.get_all_relations()
    relation_dict = dict(zip(df_relation["foodatlas_id"].tolist(), df_relation["name"].tolist()))

    df_evidence = df_evidence.groupby("triple")["source"].apply(list).reset_index()
    df_evidence["relation"] = df_evidence["triple"].apply(lambda x: relation_dict[x.split(',')[1]])

    for idx, s in enumerate(sources):
        index = (df_evidence["source"].apply(lambda x: s in x))
        index.name = s
        if idx == 0:
            df_evidence = df_evidence.set_index(index)
        else:
            df_evidence = df_evidence.set_index(index, append=True)

    upset = UpSet(
        df_evidence,
        show_counts='%d',
        sort_by='cardinality',
        intersection_plot_elements=0,
        totals_plot_elements=3,
    )
    upset.add_stacked_bars(
        by="relation",
        colors=cm.Pastel1,
        title="Number of unique triples",
        elements=7,
    )
    axes = upset.plot()
    axes["extra0"].set_yscale("log")
    plt.savefig(os.path.join(OUTPUT_DIR, "upset_plot.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "upset_plot.svg"))

    # print(df_evidence)
    # # FoodAtlas:annotation only
    # df_subset = df_evidence.loc[(False, False, True, False, False, False, False), :]
    # print(df_subset.shape)
    # df_subset_contains = df_subset[df_subset["relation"] == "contains"]
    # print(df_subset_contains.shape)


def upset_plot_contains(fa_kg):
    df_evidence = fa_kg.get_evidence()
    contains_foodatlas_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_evidence = df_evidence[df_evidence["relation"].apply(lambda x: x == contains_foodatlas_id)]
    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:prediction:entailment', 'FoodAtlas')
    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:annotation', 'FoodAtlas')
    df_evidence["source"] = df_evidence["source"].str.replace(
        'FoodAtlas:prediction:lp', 'FoodAtlas')
    df_evidence["source"] = df_evidence.apply(
        lambda row: f"{row['source']} ({row['quality']})", axis=1)
    sources = sorted(list(set(df_evidence['source'].tolist())))

    df_relation = fa_kg.get_all_relations()
    relation_dict = dict(zip(df_relation["foodatlas_id"].tolist(), df_relation["name"].tolist()))

    df_evidence = df_evidence.groupby("triple")["source"].apply(list).reset_index()
    df_evidence["relation"] = df_evidence["triple"].apply(lambda x: relation_dict[x.split(',')[1]])

    for idx, s in enumerate(sources):
        index = (df_evidence["source"].apply(lambda x: s in x))
        index.name = s
        if idx == 0:
            df_evidence = df_evidence.set_index(index)
        else:
            df_evidence = df_evidence.set_index(index, append=True)

    upset = UpSet(
        df_evidence,
        show_counts='%d',
        sort_by='cardinality',
        intersection_plot_elements=0,
        totals_plot_elements=3,
    )
    upset.add_stacked_bars(
        by="relation",
        colors=cm.Pastel1,
        title="Number of unique triples",
        elements=7,
    )
    axes = upset.plot()
    axes["extra0"].set_yscale("log")
    plt.savefig(os.path.join(OUTPUT_DIR, "upset_plot_contains.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "upset_plot_contains.svg"))

    # print(df_evidence)
    # # FoodAtlas:annotation only
    # df_subset = df_evidence.loc[(False, False, True, False, False, False, False), :]
    # print(df_subset.shape)
    # df_subset_contains = df_subset[df_subset["relation"] == "contains"]
    # print(df_subset_contains.shape)


def visualize_entities(fa_kg):
    df_entities = fa_kg.get_all_entities()
    df_evidence = fa_kg.get_evidence()

    print(df_entities)
    print(df_evidence)

    contains_foodatlas_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]

    # chemicals
    df_chemicals = df_entities[df_entities["type"].apply(
        lambda x: x == "chemical" or x.startswith("chemical:"))]
    print(f"Number of chemical entities: {df_chemicals.shape[0]}")

    df_kg_contains = fa_kg.get_kg_using_relation_name("contains")
    contains_chemicals = set(df_kg_contains["tail"].tolist())

    df_chemicals_mesh = df_chemicals[df_chemicals["foodatlas_id"].apply(
        lambda x: x not in contains_chemicals)]
    print(f"Number of chemicals used only for MeSH ontology: {df_chemicals_mesh.shape[0]}")

    df_evidence_fa = df_evidence[df_evidence["source"].apply(
        lambda x: x.startswith("FoodAtlas"))]
    df_evidence_non_fa = df_evidence[df_evidence["source"].apply(
        lambda x: not x.startswith("FoodAtlas"))]

    df_evidence_fa_contains = df_evidence_fa[df_evidence_fa["relation"].apply(
        lambda x: x == contains_foodatlas_id)]
    df_evidence_non_fa_contains = df_evidence_non_fa[df_evidence_non_fa["relation"].apply(
        lambda x: x == contains_foodatlas_id)]

    fa_contains_chemicals = set(df_evidence_fa_contains["tail"].tolist())
    non_fa_contains_chemicals = set(df_evidence_non_fa_contains["tail"].tolist())

    df_chemicals_only_ours = df_chemicals[df_chemicals["foodatlas_id"].apply(
        lambda x: x in fa_contains_chemicals and x not in non_fa_contains_chemicals)]
    print(f"Number of chemicals used only by us: {df_chemicals_only_ours.shape[0]}")

    df_chemicals_only_ext = df_chemicals[df_chemicals["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_chemicals and x not in fa_contains_chemicals)]
    print(f"Number of chemicals used only by external: {df_chemicals_only_ext.shape[0]}")

    df_chemicals_both = df_chemicals[df_chemicals["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_chemicals and x in fa_contains_chemicals)]
    print(f"Number of chemicals only by both external and us: {df_chemicals_both.shape[0]}")

    # foods
    df_foods = df_entities[df_entities["type"].apply(
        lambda x: x == "organism" or x.startswith("organism:"))]
    print(f"Number of food entities: {df_foods.shape[0]}")

    contains_foods = set(df_kg_contains["head"].tolist())

    df_foods_taxonomy = df_foods[df_foods["foodatlas_id"].apply(
        lambda x: x not in contains_foods)]
    print(f"Number of foods used only for NCBI taxonomy: {df_foods_taxonomy.shape[0]}")

    fa_contains_foods = set(df_evidence_fa_contains["head"].tolist())
    non_fa_contains_foods = set(df_evidence_non_fa_contains["head"].tolist())

    df_foods_only_ours = df_foods[df_foods["foodatlas_id"].apply(
        lambda x: x in fa_contains_foods and x not in non_fa_contains_foods)]
    print(f"Number of foods used only by us: {df_foods_only_ours.shape[0]}")

    df_foods_only_ext = df_foods[df_foods["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_foods and x not in fa_contains_foods)]
    print(f"Number of foods used only by external: {df_foods_only_ext.shape[0]}")

    df_foods_both = df_foods[df_foods["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_foods and x in fa_contains_foods)]
    print(f"Number of foods only by both external and us: {df_foods_both.shape[0]}")

    # foods with part
    df_foods_with_part = df_entities[df_entities["type"].apply(
        lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]
    print(f"Number of food - part entities: {df_foods_with_part.shape[0]}")

    contains_foods = set(df_kg_contains["head"].tolist())

    df_foods_with_part_taxonomy = df_foods_with_part[df_foods_with_part["foodatlas_id"].apply(
        lambda x: x not in contains_foods)]
    print("Number of foods_with_part used only for NCBI taxonomy: "
          f"{df_foods_with_part_taxonomy.shape[0]}")

    df_foods_with_part_only_ours = df_foods_with_part[df_foods_with_part["foodatlas_id"].apply(
        lambda x: x in fa_contains_foods and x not in non_fa_contains_foods)]
    print(f"Number of foods_with_part used only by us: {df_foods_with_part_only_ours.shape[0]}")

    df_foods_with_part_only_ext = df_foods_with_part[df_foods_with_part["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_foods and x not in fa_contains_foods)]
    print("Number of foods_with_part used only by external: "
          f"{df_foods_with_part_only_ext.shape[0]}")

    df_foods_with_part_both = df_foods_with_part[df_foods_with_part["foodatlas_id"].apply(
        lambda x: x in non_fa_contains_foods and x in fa_contains_foods)]
    print("Number of foods_with_part only by both external and us: "
          f"{df_foods_with_part_both.shape[0]}")

    data = [
        # food
        ["Organism (food)", "Taxonomic lineage", df_foods_taxonomy.shape[0]],
        ["Organism (food)", "FoodAtlas only", df_foods_only_ours.shape[0]],
        ["Organism (food)", "External DB only", df_foods_only_ext.shape[0]],
        ["Organism (food)", "FoodAtlas & external db", df_foods_both.shape[0]],
        # food - part
        ["Food - part", "FoodAtlas only", df_foods_with_part_only_ours.shape[0]],
        # chemical
        ["Chemical", "MeSH", df_chemicals_mesh.shape[0]],
        ["Chemical", "FoodAtlas only", df_chemicals_only_ours.shape[0]],
        ["Chemical", "External DB only", df_chemicals_only_ext.shape[0]],
        ["Chemical", "FoodAtlas & external db", df_chemicals_both.shape[0]],
    ]

    df = pd.DataFrame(data, columns=["Entity Type", "Source", "Count"])
    print(df)

    category_orders = {
        "Source": [
            "External DB only",
            "FoodAtlas & external db",
            "MeSH",
            "Taxonomic lineage",
            "FoodAtlas only",
        ],
    }

    fig = px.bar(
        df,
        x="Entity Type",
        y="Count",
        color="Source",
        category_orders=category_orders,
        text="Count",
        width=500,
        height=400,
    )

    fig.update_traces(width=0.5)

    subplots = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        row_heights=[0.25, 0.75],
    )

    for d in fig.data:
        subplots.append_trace(d, row=1, col=1)
        d.showlegend = False
        subplots.append_trace(d, row=2, col=1)

    subplots.update_yaxes(range=[20000, 22000], row=1, col=1)
    subplots.update_xaxes(visible=False, row=1, col=1)
    subplots.update_yaxes(range=[0, 5000], row=2, col=1)
    subplots.update_layout(
        barmode='stack',
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        xaxis={
            'categoryorder': 'array',
            'categoryarray': ["Organism (food)", "Food - part", "Chemical"]
        },
    )
    subplots.write_image(os.path.join(OUTPUT_DIR, "entities.png"))
    subplots.write_image(os.path.join(OUTPUT_DIR, "entities.svg"))


def visualize_relations(fa_kg):
    # relations
    df_evidence = fa_kg.get_evidence()
    df_evidence = df_evidence.groupby("triple")["quality"].apply(list).reset_index()

    def _select_quality(x):
        if "high" in x:
            return "high"
        elif "medium" in x:
            return "medium"
        else:
            return "low"
    df_evidence["quality"] = df_evidence["quality"].apply(_select_quality)

    print(f"Number of all triples: {df_evidence.shape[0]}")

    isA_fa_id = fa_kg.get_relation_by_name("isA")["foodatlas_id"]
    df_isA = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == isA_fa_id)]
    print(f"Number of isA triples: {df_isA.shape[0]}")

    hasChild_fa_id = fa_kg.get_relation_by_name("hasChild")["foodatlas_id"]
    df_hasChild = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == hasChild_fa_id)]
    print(f"Number of hasChild triples: {df_hasChild.shape[0]}")

    hasPart_fa_id = fa_kg.get_relation_by_name("hasPart")["foodatlas_id"]
    df_hasPart = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == hasPart_fa_id)]
    print(f"Number of hasPart triples: {df_hasPart.shape[0]}")

    df_hasPart_high = df_hasPart[df_hasPart["quality"] == "high"]
    print(f"Number of hasPart high quality triples: {df_hasPart_high.shape[0]}")
    df_hasPart_medium = df_hasPart[df_hasPart["quality"] == "medium"]
    print(f"Number of hasPart medium quality triples: {df_hasPart_medium.shape[0]}")
    df_hasPart_low = df_hasPart[df_hasPart["quality"] == "low"]
    print(f"Number of hasPart low quality triples: {df_hasPart_low.shape[0]}")

    contains_fa_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_contains = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == contains_fa_id)]
    print(f"Number of contains triples: {df_contains.shape[0]}")

    df_contains_high = df_contains[df_contains["quality"] == "high"]
    print(f"Number of contains high quality triples: {df_contains_high.shape[0]}")
    df_contains_medium = df_contains[df_contains["quality"] == "medium"]
    print(f"Number of contains medium quality triples: {df_contains_medium.shape[0]}")
    df_contains_low = df_contains[df_contains["quality"] == "low"]
    print(f"Number of contains low quality triples: {df_contains_low.shape[0]}")

    data = [
        # contains
        ["contains - High", df_contains_high.shape[0]],
        ["contains - Medium", df_contains_medium.shape[0]],
        ["contains - Low", df_contains_low.shape[0]],
        # isA
        ["isA - Medium", df_isA.shape[0]],
        # hasChild
        ["hasChild - Medium", df_hasChild.shape[0]],
        # hasPart
        ["hasPart - High", df_hasPart_high.shape[0]],
        ["hasPart - Medium", df_hasPart_medium.shape[0]],
    ]

    df = pd.DataFrame(data, columns=["Relation", "Count"])
    print(df)

    category_orders = {
        "Relation": [
            "contains - High",
            "contains - Medium",
            "contains - Low",
            "hasPart - High",
            "hasPart - Medium",
            "isA - Medium",
            "hasChild - Medium",
        ],
    }

    fig = px.bar(
        df,
        x="Relation",
        y="Count",
        log_y=True,
        category_orders=category_orders,
        text="Count",
        width=500,
        height=400,
    )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "relations.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "relations.svg"))


def plot_sunburst_chemicals(chemicals_group_filename):
    df_chemicals = pd.read_csv(chemicals_group_filename, sep='\t', keep_default_na=False)
    df_chemicals = df_chemicals[df_chemicals["is_contains"]]
    print(df_chemicals)

    def _custom_wrap(s, width=20):
        return "<br>".join(textwrap.wrap(s, width=width))
    df_chemicals["group"] = df_chemicals["group"].map(_custom_wrap)
    df_chemicals["subgroup"] = df_chemicals["subgroup"].map(_custom_wrap)

    fig = px.sunburst(
        df_chemicals,
        path=['group', 'subgroup'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_chemicals.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_chemicals.svg"))


def plot_sunburst_organisms(organisms_group_filename):
    df_organisms = pd.read_csv(organisms_group_filename, sep='\t', keep_default_na=False)
    df_organisms = df_organisms[df_organisms["is_food"]]
    df_organisms["group"] = df_organisms["group"].apply(
        lambda x: "Unknown" if x == "" else x)
    df_organisms["subgroup"] = df_organisms["subgroup"].apply(
        lambda x: "Unknown" if x == "" else x)
    print(df_organisms)

    def _custom_wrap(s, width=20):
        return "<br>".join(textwrap.wrap(s, width=width))
    df_organisms["group"] = df_organisms["group"].map(_custom_wrap)
    df_organisms["subgroup"] = df_organisms["subgroup"].map(_custom_wrap)

    fig = px.sunburst(
        df_organisms,
        path=['group', 'subgroup'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_foods.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_foods.svg"))


def plot_treemap_chemicals(chemicals_group_filename):
    df_chemicals = pd.read_csv(chemicals_group_filename, sep='\t', keep_default_na=False)
    df_chemicals = df_chemicals[df_chemicals["is_contains"]]
    print(df_chemicals)

    fig = px.treemap(
        df_chemicals,
        path=['group', 'subgroup'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_layout(
        uniformtext=dict(minsize=100),
        margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "treemap_chemicals.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "treemap_chemicals.svg"))


def plot_treemap_organisms(organisms_group_filename):
    df_organisms = pd.read_csv(organisms_group_filename, sep='\t', keep_default_na=False)
    df_organisms = df_organisms[df_organisms["is_food"]]
    df_organisms["group"] = df_organisms["group"].apply(
        lambda x: "Unknown" if x == "" else x)
    df_organisms["subgroup"] = df_organisms["subgroup"].apply(
        lambda x: "Unknown" if x == "" else x)
    print(df_organisms)

    fig = px.treemap(
        df_organisms,
        path=['group', 'subgroup'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "treemap_foods.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "treemap_foods.svg"))


def plot_num_sources_per_triple(fa_kg):
    df_evidence = fa_kg.get_evidence()
    contains_fa_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_evidence = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == contains_fa_id)]
    df_evidence = df_evidence.groupby("triple")["source"].apply(list).reset_index()
    df_evidence["num_source"] = df_evidence["source"].apply(len)

    num_sources = df_evidence["num_source"].tolist()
    num_sources_avg = np.mean(num_sources)
    num_sources_std = np.std(num_sources)
    print(f"Number of sources: {num_sources_avg}+-{num_sources_std}")
    print("Percentage of sources with num_source == 1: "
          f"{num_sources.count(1)} out of {len(num_sources)}")

    df = df_evidence.groupby("num_source")["triple"].apply(len).reset_index()
    df.rename({"num_source": "# of sources", "triple": "# of triples"}, inplace=True, axis=1)
    print(df)

    fig = px.bar(
        df,
        x="# of sources",
        y="# of triples",
        log_y=True,
        width=800,
        height=400,
    )
    fig.update_layout(
        margin=dict(t=0, l=2, r=2, b=0),
        font_family="Arial",
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "num_sources_per_triple.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "num_sources_per_triple.svg"))


def generate_kg(fa_kg):
    df_kg = fa_kg.get_kg()
    G = nx.DiGraph()
    for _, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        G.add_edge(str(row["head"]), str(row["tail"]))

    # create entity lookup dict
    df_entities = fa_kg.get_all_entities()
    dictionary = dict(zip(df_entities["foodatlas_id"], df_entities["name"]))

    return G, dictionary


def plot_node_stats(fa_kg, G, dictionary):
    # degree centrality
    # The number of edges it has (either in or out).
    # Higher = more central.
    print('Degree centrality')
    degree_centrality = nx.degree_centrality(G)
    df_degree_centrality = pd.DataFrame({
        "foodatlas_id": degree_centrality.keys(),
        "degree_centrality": degree_centrality.values(),
    })
    df_degree_centrality.sort_values("degree_centrality", ascending=False, inplace=True)

    # betweenness centrality
    # The number of shortest paths between all pair
    # of nodes that pass through the node of interest.
    # Higher = More control over the network
    # since more information passes through this node.
    print('betweenness centrality')
    betweenness_centrality = nx.betweenness_centrality(G)
    df_betweenness_centrality = pd.DataFrame({
        "foodatlas_id": betweenness_centrality.keys(),
        "betweenness_centrality": betweenness_centrality.values(),
    })
    df_betweenness_centrality.sort_values("betweenness_centrality", ascending=False, inplace=True)

    # closeness centrality
    # Average length of the shortest path between the
    # node and all other nodes in the graph.
    # Higher = close to all other nodes
    print('closeness centrality')
    closeness_centrality = nx.closeness_centrality(G)
    df_closeness_centrality = pd.DataFrame({
        "foodatlas_id": closeness_centrality.keys(),
        "closeness_centrality": closeness_centrality.values(),
    })
    df_closeness_centrality.sort_values("closeness_centrality", ascending=False, inplace=True)

    # eigenvector centrality
    print('eigenvector centrality')
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    df_eigenvector_centrality = pd.DataFrame({
        "foodatlas_id": eigenvector_centrality.keys(),
        "eigenvector_centrality": eigenvector_centrality.values(),
    })
    df_eigenvector_centrality.sort_values("eigenvector_centrality", ascending=False, inplace=True)

    df_degree_centrality["name"] = df_degree_centrality["foodatlas_id"].apply(
        lambda x: dictionary[x])
    df_betweenness_centrality["name"] = df_betweenness_centrality["foodatlas_id"].apply(
        lambda x: dictionary[x])
    df_closeness_centrality["name"] = df_closeness_centrality["foodatlas_id"].apply(
        lambda x: dictionary[x])
    df_eigenvector_centrality["name"] = df_eigenvector_centrality["foodatlas_id"].apply(
        lambda x: dictionary[x])

    df_degree_centrality["type"] = df_degree_centrality["foodatlas_id"].apply(
        lambda x: fa_kg.get_entity_by_id(x)["type"].split(':')[0])
    df_betweenness_centrality["type"] = df_betweenness_centrality["foodatlas_id"].apply(
        lambda x: fa_kg.get_entity_by_id(x)["type"].split(':')[0])
    df_closeness_centrality["type"] = df_closeness_centrality["foodatlas_id"].apply(
        lambda x: fa_kg.get_entity_by_id(x)["type"].split(':')[0])
    df_eigenvector_centrality["type"] = df_eigenvector_centrality["foodatlas_id"].apply(
        lambda x: fa_kg.get_entity_by_id(x)["type"].split(':')[0])

    df_degree_centrality.to_csv(
        os.path.join(OUTPUT_DIR, "df_degree_centrality.txt"), sep='\t', index=False)
    df_betweenness_centrality.to_csv(
        os.path.join(OUTPUT_DIR, "df_betweenness_centrality.txt"), sep='\t', index=False)
    df_closeness_centrality.to_csv(
        os.path.join(OUTPUT_DIR, "df_closeness_centrality.txt"), sep='\t', index=False)
    df_eigenvector_centrality.to_csv(
        os.path.join(OUTPUT_DIR, "df_eigenvector_centrality.txt"), sep='\t', index=False)

    print(df_degree_centrality)
    print(df_betweenness_centrality)
    print(df_closeness_centrality)
    print(df_eigenvector_centrality)

    df_degree_centrality = df_degree_centrality.head(20)
    df_betweenness_centrality = df_betweenness_centrality.head(20)
    df_closeness_centrality = df_closeness_centrality.head(20)
    df_eigenvector_centrality = df_eigenvector_centrality.head(20)

    fig = make_subplots(rows=1, cols=2)

    fig1 = px.bar(
        df_degree_centrality,
        x='name', y='degree_centrality',
    )

    fig2 = px.bar(
        df_closeness_centrality,
        x='name', y='closeness_centrality',
    )

    for d in fig1.data:
        fig.add_trace((go.Bar(x=d['x'], y=d['y'], name=d['name'])), row=1, col=1)

    for d in fig2.data:
        fig.add_trace((go.Bar(x=d['x'], y=d['y'], name=d['name'])), row=1, col=2)

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        height=225, width=1000,
    )
    fig.update_xaxes(tickangle=45)
    fig.write_image(os.path.join(OUTPUT_DIR, "centrality.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "centrality.svg"))


def visualize_sankey(fa_kg):
    df_evidence = fa_kg.get_evidence()
    print(f"Evidence shape: {df_evidence.shape[0]}")
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()

    opacity = 0.4
    color_dict = {
        # source
        'FDC': '#dbecff',
        'FoodAtlas': '#dcf0cb',
        'Frida': '#ffeb9c',
        'MeSH': '#48e797',
        'NCBI_taxonomy': '#2be959',
        'Phenol-Explorer': '#7ffde9',
        # quality
        'high': '#666666',
        'low': '#e6e6e6',
        'medium': '#b3b3b3',
        # pmid / pmcid
        'pmid': '#fd8c80',
        'none': '#dfef59',
        'pmid & pmcid': '#ff6343',
    }

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    color_dict = {k: hex_to_rgb(v) for k, v in color_dict.items()}

    def _check_pmid_pmcid(row):
        if row["pmid"] != "" and row["pmcid"]!= "":
            return "pmid & pmcid"
        elif row["pmid"] != "":
            return "pmid"
        elif row["pmcid"] != "":
            return "pmcid"
        else:
            return "none"
    df_evidence["pmid/pmcid"] = df_evidence.apply(lambda row: _check_pmid_pmcid(row), axis=1)

    # node
    quality_label = list(set(df_evidence["quality"]))
    source_label = list(set(
        [x if not x.startswith("FoodAtlas") else "FoodAtlas" for x in df_evidence["source"]]))
    pmid_pmcid_label = list(set(df_evidence["pmid/pmcid"]))
    label = sorted(quality_label + source_label + pmid_pmcid_label)
    label = {x: idx for idx, x in enumerate(label)}
    node_color = [f"rgba{color_dict[x] + (1.0,)}" for x in list(label.keys())]

    # link
    source = []
    target = []
    value = []

    # quality - source
    for q, s in sorted(list(product(quality_label, source_label))):
        df_temp = df_evidence[df_evidence["quality"] == q]
        df_temp = df_temp[df_temp["source"].apply(lambda x: x.startswith(s))]
        size = df_temp.shape[0]
        if size > 0:
            source.append(label[q])
            target.append(label[s])
            value.append(np.log2(size))

    # source - pmid/pmcid
    for s, p in sorted(list(product(source_label, pmid_pmcid_label))):
        df_temp = df_evidence[df_evidence["source"].apply(lambda x: x.startswith(s))]
        df_temp = df_temp[df_temp["pmid/pmcid"] == p]
        size = df_temp.shape[0]
        if size > 0:
            source.append(label[s])
            target.append(label[p])
            value.append(np.log2(size))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=list(label.keys()),
            color=node_color,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        )
    )])

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        width=600,
        height=300,
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "sankey.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sankey.svg"))


def venn_diagram(fa_kg):
    import matplotlib
    matplotlib.use('Agg')
    from venn import venn

    df_evidence = fa_kg.get_evidence()
    contains_foodatlas_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_evidence = df_evidence[df_evidence["relation"].apply(lambda x: x == contains_foodatlas_id)]
    print(f"Number of df_evidence: {df_evidence.shape[0]}")

    df_fa = df_evidence[df_evidence["source"].apply(lambda x: x.startswith("FoodAtlas"))]
    print(f"Number of df_fa: {df_fa.shape[0]}")
    df_frida = df_evidence[df_evidence["source"].apply(lambda x: x == "Frida")]
    print(f"Number of df_frida: {df_frida.shape[0]}")
    df_phenol_explorer = df_evidence[df_evidence["source"].apply(lambda x: x == "Phenol-Explorer")]
    print(f"Number of df_phenol_explorer: {df_phenol_explorer.shape[0]}")
    df_fdc = df_evidence[df_evidence["source"].apply(lambda x: x == "FDC")]
    print(f"Number of df_fdc: {df_fdc.shape[0]}")

    # high
    print()
    df_fa_high = df_fa[df_fa["quality"] == "high"]
    print(f"Number of df_fa_high: {df_fa_high.shape[0]}")
    df_frida_high = df_frida[df_frida["quality"] == "high"]
    print(f"Number of df_frida_high: {df_frida_high.shape[0]}")
    df_phenol_explorer_high = df_phenol_explorer[df_phenol_explorer["quality"] == "high"]
    print(f"Number of df_phenol_explorer_high: {df_phenol_explorer_high.shape[0]}")
    df_fdc_high = df_fdc[df_fdc["quality"] == "high"]
    print(f"Number of df_fdc_high: {df_fdc_high.shape[0]}")

    # medium
    print()
    df_fa_medium = df_fa[df_fa["quality"] == "medium"]
    print(f"Number of df_fa_medium: {df_fa_medium.shape[0]}")
    df_frida_medium = df_frida[df_frida["quality"] == "medium"]
    print(f"Number of df_frida_medium: {df_frida_medium.shape[0]}")
    df_phenol_explorer_medium = df_phenol_explorer[df_phenol_explorer["quality"] == "medium"]
    print(f"Number of df_phenol_explorer_medium: {df_phenol_explorer_medium.shape[0]}")
    df_fdc_medium = df_fdc[df_fdc["quality"] == "medium"]
    print(f"Number of df_fdc_medium: {df_fdc_medium.shape[0]}")

    # low
    print()
    df_fa_low = df_fa[df_fa["quality"] == "low"]
    print(f"Number of df_fa_low: {df_fa_low.shape[0]}")
    df_frida_low = df_frida[df_frida["quality"] == "low"]
    print(f"Number of df_frida_low: {df_frida_low.shape[0]}")
    df_phenol_explorer_low = df_phenol_explorer[df_phenol_explorer["quality"] == "low"]
    print(f"Number of df_phenol_explorer_low: {df_phenol_explorer_low.shape[0]}")
    df_fdc_low = df_fdc[df_fdc["quality"] == "low"]
    print(f"Number of df_fdc_low: {df_fdc_low.shape[0]}")

    # high
    data = {
        "df_fa_high": set(df_fa_high["triple"].tolist()),
        "df_frida_high": set(df_frida_high["triple"].tolist()),
        "df_phenol_explorer_high": set(df_phenol_explorer_high["triple"].tolist()),
        "df_fdc_high": set(df_fdc_high["triple"].tolist()),
    }

    ax = venn(data)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_high.png"))
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_high.svg"))

    # medium
    data = {
        "df_fa_medium": set(df_fa_medium["triple"].tolist()),
        "df_frida_medium": set(df_frida_medium["triple"].tolist()),
        "df_phenol_explorer_medium": set(df_phenol_explorer_medium["triple"].tolist()),
        # "df_fdc_medium": set(df_fdc_medium["triple"].tolist()),
    }

    ax = venn(data)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_medium.png"))
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_medium.svg"))

    # low
    data = {
        # "df_fa_low": set(df_fa_low["triple"].tolist()),
        "df_frida_low": set(df_frida_low["triple"].tolist()),
        # "df_phenol_explorer_low": set(df_phenol_explorer_low["triple"].tolist()),
        "df_fdc_low": set(df_fdc_low["triple"].tolist()),
    }

    ax = venn(data)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_low.png"))
    fig.savefig(os.path.join(OUTPUT_DIR, "venn_low.svg"))

    # low
    data = {
        "df_fa_high": set(df_fa_high["triple"].tolist()),
        "df_fa_medium": set(df_fa_medium["triple"].tolist()),
        "df_frida": set(df_frida["triple"].tolist()),
        "df_phenol_explorer": set(df_phenol_explorer["triple"].tolist()),
        "df_fdc": set(df_fdc["triple"].tolist()),
    }

    ax = venn(data)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_DIR, "venn.png"))
    fig.savefig(os.path.join(OUTPUT_DIR, "venn.svg"))


def count_high_med_low_quality_triples(fa_kg):
    df_evidence = fa_kg.get_evidence()
    df_evidence = df_evidence.groupby("triple")["quality"].apply(list).reset_index()
    print(df_evidence.shape[0])

    def _select_quality(x):
        if "high" in x:
            return "high"
        elif "medium" in x:
            return "medium"
        else:
            return "low"
    df_evidence["quality"] = df_evidence["quality"].apply(_select_quality)

    df_high = df_evidence[df_evidence["quality"] == "high"]
    df_medium = df_evidence[df_evidence["quality"] == "medium"]
    df_low = df_evidence[df_evidence["quality"] == "low"]

    print(f"Number of high-quality triples: {df_high.shape[0]}")
    print(f"Number of medium-quality triples: {df_medium.shape[0]}")
    print(f"Number of low-quality triples: {df_low.shape[0]}")


def count_fig3a(fa_kg):
    df_kg = fa_kg.get_kg()
    df_relations = fa_kg.get_all_relations()
    df_entities =fa_kg.get_all_entities()

    rel_dict = dict(zip(df_relations['name'], df_relations['foodatlas_id']))
    print(rel_dict)

    relations = df_kg['relation'].tolist()
    print(Counter(relations))

    df_entities_foodpart = fa_kg.get_entities_by_type(
        exact_type='organism_with_part',
        startswith_type='organism_with_part:',
    )
    print(df_entities_foodpart)

    df_entities_food = fa_kg.get_entities_by_type(
        exact_type='organism',
        startswith_type='organism:',
    )
    print(df_entities_food)

    df_entities_chemical = fa_kg.get_entities_by_type(
        exact_type='chemical',
        startswith_type='chemical:',
    )
    print(df_entities_chemical)

    df_contains = df_kg[df_kg['relation'] == rel_dict['contains']]
    print(df_contains)

    df_foodpart_contains = df_contains[df_contains['head'].apply(
        lambda x: x in df_entities_foodpart['foodatlas_id'].tolist()
    )]
    print(df_foodpart_contains)


def main():
    fa_kg = KnowledgeGraph(kg_dir=FINAL_DATA_DIR)

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # count_fig3a(fa_kg)
    # visualize_source(fa_kg)
    # visualize_entities(fa_kg)
    # visualize_relations(fa_kg)
    # upset_plot(fa_kg)
    # upset_plot_contains(fa_kg)
    plot_sunburst_chemicals("../../outputs/backend_data/v0.1/chemicals_group.txt")
    plot_sunburst_organisms("../../outputs/backend_data/v0.1/organisms_group.txt")
    # plot_treemap_chemicals("../../outputs/backend_data/v0.1/chemicals_group.txt")
    # plot_treemap_organisms("../../outputs/backend_data/v0.1/organisms_group.txt")
    # plot_num_sources_per_triple(fa_kg)
    # visualize_sankey(fa_kg)
    # venn_diagram(fa_kg)
    # count_high_med_low_quality_triples(fa_kg)

    # G, dictionary = generate_kg(fa_kg)
    # plot_node_stats(fa_kg, G, dictionary)


if __name__ == "__main__":
    main()
