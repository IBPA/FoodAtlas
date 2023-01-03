import os
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
OUTPUT_DIR = "../../outputs/analysis_codes"
BACKEND_DATA_DIR = "../../outputs/backend_data/v0.1"


def pie_chart_sources(fa_kg):
    df_evidence = fa_kg.get_evidence()
    print(df_evidence)
    sources = list(set(df_evidence['source'].tolist()))
    print(f"Sources: {sources}")
    qualities = list(set(df_evidence['quality'].tolist()))
    print(f"Qualities: {qualities}")

    df_high = df_evidence[df_evidence["quality"].apply(
        lambda x: x == "high")]
    print(f"Number of df_high: {df_high.shape[0]}")

    df_medium = df_evidence[df_evidence["quality"].apply(
        lambda x: x == "medium")]
    print(f"Number of df_medium: {df_medium.shape[0]}")

    df_low = df_evidence[df_evidence["quality"].apply(
        lambda x: x == "low")]
    print(f"Number of df_low: {df_low.shape[0]}")

    df_annotation = df_evidence[df_evidence["source"].apply(
        lambda x: x == "FoodAtlas:annotation")]
    print(f"Number of df_annotation: {df_annotation.shape[0]}")

    df_prediction = df_evidence[df_evidence["source"].apply(
        lambda x: x.startswith("FoodAtlas:prediction"))]
    print(f"Number of df_prediction: {df_prediction.shape[0]}")

    df_mesh = df_evidence[df_evidence["source"].apply(lambda x: x == "MeSH")]
    print(f"Number of df_mesh: {df_mesh.shape[0]}")

    df_ncbi = df_evidence[df_evidence["source"].apply(lambda x: x == "NCBI_taxonomy")]
    print(f"Number of df_ncbi: {df_ncbi.shape[0]}")

    df_frida = df_evidence[df_evidence["source"].apply(lambda x: x == "Frida")]
    print(f"Number of df_frida: {df_frida.shape[0]}")
    df_frida_medium = df_frida[df_frida["quality"].apply(lambda x: x == "medium")]
    df_frida_low = df_frida[df_frida["quality"].apply(lambda x: x == "low")]
    print(f"Number of df_frida_medium: {df_frida_medium.shape[0]}")
    print(f"Number of df_frida_low: {df_frida_low.shape[0]}")

    df_phenol_explorer = df_evidence[df_evidence["source"].apply(lambda x: x == "Phenol-Explorer")]
    print(f"Number of df_phenol_explorer: {df_phenol_explorer.shape[0]}")
    df_phenol_explorer_medium = df_phenol_explorer[df_phenol_explorer["quality"].apply(
        lambda x: x == "medium")]
    df_phenol_explorer_low = df_phenol_explorer[df_phenol_explorer["quality"].apply(
        lambda x: x == "low")]
    print(f"Number of df_phenol_explorer_medium: {df_phenol_explorer_medium.shape[0]}")
    print(f"Number of df_phenol_explorer_low: {df_phenol_explorer_low.shape[0]}")

    rows = [
        ["Source", "", df_evidence.shape[0]],
        # High
        ["High", "Source", df_high.shape[0]],
        ["FoodAtlas:annotation", "High", df_annotation.shape[0]],
        ["MeSH", "High", df_mesh.shape[0]],
        ["NCBI", "High", df_ncbi.shape[0]],
        # Medium
        ["Medium", "Source", df_medium.shape[0]],
        ["Frida:Medium", "Medium", df_frida_medium.shape[0]],
        ["Phenol-Explorer:Medium", "Medium", df_phenol_explorer_medium.shape[0]],
        # Low
        ["Low", "Source", df_low.shape[0]],
        ["FoodAtlas:prediction", "Low", df_prediction.shape[0]],
        ["Frida:Low", "Low", df_frida_low.shape[0]],
        ["Phenol-Explorer:Low", "Low", df_phenol_explorer_low.shape[0]],
    ]

    df = pd.DataFrame(rows, columns=["labels", "parents", "values"])

    print(df["labels"])

    fig = go.Figure(go.Sunburst(
        labels=df["labels"],
        parents=df["parents"],
        values=df["values"],
        branchvalues="total",
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_sources.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_sources.svg"))


def upset_plot(fa_kg):
    df_evidence = fa_kg.get_evidence()
    sources = list(set(df_evidence['source'].tolist()))

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
        # show_percentages=True,
        intersection_plot_elements=0,
        totals_plot_elements=5,
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


def pie_chart_entities(fa_kg):
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

    rows = [
        ["Entity", "", df_entities.shape[0]],
        # food
        ["organism", "Entity", df_foods.shape[0]],
        ["food - taxonomy", "organism", df_foods_taxonomy.shape[0]],
        ["food - only us", "organism", df_foods_only_ours.shape[0]],
        ["food - only ext", "organism", df_foods_only_ext.shape[0]],
        ["food - both", "organism", df_foods_both.shape[0]],
        # food - part
        ["food-part", "Entity", df_foods_with_part.shape[0]],
        ["food-part - taxonomy", "food-part", df_foods_with_part_taxonomy.shape[0]],
        ["food-part - only us", "food-part", df_foods_with_part_only_ours.shape[0]],
        ["food-part - only ext", "food-part", df_foods_with_part_only_ext.shape[0]],
        ["food-part - both", "food-part", df_foods_with_part_both.shape[0]],
        # chemical
        ["chemical", "Entity", df_chemicals.shape[0]],
        ["chemical - MeSH", "chemical", df_chemicals_mesh.shape[0]],
        ["chemical - only us", "chemical", df_chemicals_only_ours.shape[0]],
        ["chemical - only ext", "chemical", df_chemicals_only_ext.shape[0]],
        ["chemical - both", "chemical", df_chemicals_both.shape[0]],
    ]

    df = pd.DataFrame(rows, columns=["labels", "parents", "values"])

    fig = go.Figure(go.Sunburst(
        labels=df["labels"],
        parents=df["parents"],
        values=df["values"],
        branchvalues="total",
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_entities.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_entities.svg"))


def pie_chart_relations(fa_kg):
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

    rows = [
        ["Relation", "", df_evidence.shape[0]],
        # contains
        ["contains", "Relation", df_contains.shape[0]],
        ["contains - high", "contains", df_contains_high.shape[0]],
        ["contains - medium", "contains", df_contains_medium.shape[0]],
        ["contains - low", "contains", df_contains_low.shape[0]],
        # isA
        ["isA", "Relation", df_isA.shape[0]],
        ["isA - high", "isA", df_isA.shape[0]],
        # hasChild
        ["hasChild", "Relation", df_hasChild.shape[0]],
        ["hasChild - high", "hasChild", df_hasChild.shape[0]],
        # hasPart
        ["hasPart", "Relation", df_hasPart.shape[0]],
        ["hasPart - high", "hasPart", df_hasPart_high.shape[0]],
        ["hasPart - medium", "hasPart", df_hasPart_medium.shape[0]],
        ["hasPart - low", "hasPart", df_hasPart_low.shape[0]],
    ]

    df = pd.DataFrame(rows, columns=["labels", "parents", "values"])

    fig = go.Figure(go.Sunburst(
        labels=df["labels"],
        parents=df["parents"],
        values=df["values"],
        branchvalues="total",
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_relations.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_relations.svg"))


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


def plot_num_sources_per_triple(fa_kg):
    df_evidence = fa_kg.get_evidence()
    df_evidence = df_evidence.groupby("triple")["source"].apply(list).reset_index()
    df_evidence["num_source"] = df_evidence["source"].apply(len)
    contains_fa_id = fa_kg.get_relation_by_name("contains")["foodatlas_id"]
    df_evidence = df_evidence[df_evidence["triple"].apply(
        lambda x: x.split(',')[1] == contains_fa_id)]

    num_sources = df_evidence["num_source"].tolist()
    num_sources_avg = np.mean(num_sources)
    num_sources_std = np.std(num_sources)
    print(f"Number of sources: {num_sources_avg}+-{num_sources_std}")
    print("Percentage of sources with num_source == 1: "
          f"{num_sources.count(1)} out of {len(num_sources)}")

    df = df_evidence.groupby("num_source")["triple"].apply(len).reset_index()
    df.rename({"num_source": "# of sources", "triple": "# of triples"}, inplace=True, axis=1)

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
        xaxis=dict(tickfont=dict(size=9)))
    fig.update_xaxes(tickangle=90, tickmode="linear")
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
    closeness_centrality = nx.closeness_centrality(G)
    df_closeness_centrality = pd.DataFrame({
        "foodatlas_id": closeness_centrality.keys(),
        "closeness_centrality": closeness_centrality.values(),
    })
    df_closeness_centrality.sort_values("closeness_centrality", ascending=False, inplace=True)

    # eigenvector centrality
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


def main():
    fa_kg = KnowledgeGraph(kg_dir=FINAL_DATA_DIR)

    # pie_chart_sources(fa_kg)
    # upset_plot(fa_kg)
    # pie_chart_entities(fa_kg)
    # pie_chart_relations(fa_kg)
    # plot_sunburst_chemicals("../../outputs/backend_data/v0.1/chemicals_group.txt")
    # plot_sunburst_organisms("../../outputs/backend_data/v0.1/organisms_group.txt")
    plot_num_sources_per_triple(fa_kg)

    # G, dictionary = generate_kg(fa_kg)
    # plot_node_stats(fa_kg, G, dictionary)


if __name__ == "__main__":
    main()
