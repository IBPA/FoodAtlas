# 2a. draw dotted line between food and chemical?
# 2b. remove inner circle
# 2b. we should only show chemicals and foods that are used for contains relationship
# 2c. only inner circle for contains

import os
import sys
import textwrap

sys.path.append('../data_processing/')

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from tqdm import tqdm  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

KG_DIR = "../../outputs/kg/annotations_mesh_ncbi"
OUTPUT_DIR = "../../outputs/analysis_codes"
FOOD_COLOR = "#87aadeff"
FOOD_PART_COLOR = "#afdde9ff"
CHEMICAL_COLOR = "#ffeeaaff"
HASPART_COLOR = "#338000ff"
CONTAINS_COLOR = "#d35f5fff"
HASCHILD_COLOR = "#2c5aa0ff"
ISA_COLOR = "#6c5d53ff"


def pie_chart_sources(fa_kg):
    df_evidence = fa_kg.get_evidence()
    print(df_evidence)
    sources = list(set(df_evidence['source'].tolist()))
    print(f"Sources: {sources}")

    df_annotation = df_evidence[df_evidence["source"].apply(lambda x: x.startswith("annotation"))]
    print(f"Number of df_annotation: {df_annotation.shape[0]}")

    df_mesh = df_evidence[df_evidence["source"].apply(lambda x: x == "MESH")]
    print(f"Number of df_mesh: {df_mesh.shape[0]}")

    df_ncbi = df_evidence[df_evidence["source"].apply(lambda x: x == "NCBI_taxonomy")]
    print(f"Number of df_ncbi: {df_ncbi.shape[0]}")

    rows = [
        ["Source", "", df_evidence.shape[0]],
        # annotation
        ["annotation", "Source", df_annotation.shape[0]],
        # MeSH
        ["MeSH", "Source", df_mesh.shape[0]],
        # NCBI
        ["NCBI", "Source", df_ncbi.shape[0]],
        # prediction
    ]

    df = pd.DataFrame(rows, columns=["labels", "parents", "values"])

    fig = go.Figure(go.Sunburst(
        labels=df["labels"],
        parents=df["parents"],
        values=df["values"],
        branchvalues="total",
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_sources.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_sources.svg"))


def pie_chart_entities(fa_kg):
    df_entities = fa_kg.get_all_entities()
    df_evidence = fa_kg.get_evidence()

    print(df_entities)
    print(df_evidence)

    df_chemicals = df_entities[df_entities["type"].apply(
        lambda x: x == "chemical" or x.startswith("chemical:"))]
    print(f"Number of chemical entities: {df_chemicals.shape[0]}")

    df_foods = df_entities[df_entities["type"].apply(
        lambda x: x == "organism" or x.startswith("organism:"))]
    print(f"Number of food entities: {df_foods.shape[0]}")

    df_food_with_part = df_entities[df_entities["type"].apply(
        lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]
    print(f"Number of food - part entities: {df_food_with_part.shape[0]}")

    rows = [
        ["Entity", "", df_entities.shape[0]],
        # food
        ["food", "Entity", df_foods.shape[0]],
        ["f - ours", "food", 400],
        ["f - external", "food", 400],
        ["NCBI", "food", df_foods.shape[0] - 800],
        # food - part
        ["food - part", "Entity", df_food_with_part.shape[0]],
        ["fp - ours", "food - part", 100],
        ["fp - external", "food - part", df_food_with_part.shape[0] - 100],
        # chemical
        ["chemical", "Entity", df_chemicals.shape[0]],
        ["c - ours", "chemical", 400],
        ["c - external", "chemical", 400],
        ["MeSH", "chemical", df_chemicals.shape[0] - 800],
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
    df_kg = fa_kg.get_kg()
    print(f"Number of all triples: {df_kg.shape[0]}")

    df_relations = fa_kg.get_all_relations()
    relations_lookup = dict(zip(
        df_relations["name"].tolist(), df_relations["foodatlas_id"].tolist()))

    df_isA = df_kg[df_kg["relation"] == relations_lookup["isA"]]
    print(f"Number of isA triples: {df_isA.shape[0]}")

    df_hasChild = df_kg[df_kg["relation"] == relations_lookup["hasChild"]]
    print(f"Number of hasChild triples: {df_hasChild.shape[0]}")

    df_hasPart = df_kg[df_kg["relation"] == relations_lookup["hasPart"]]
    print(f"Number of hasPart triples: {df_hasPart.shape[0]}")

    df_contains = df_kg[df_kg["relation"] == relations_lookup["contains"]]
    print(f"Number of contains triples: {df_contains.shape[0]}")

    rows = [
        ["Relation", "", df_kg.shape[0]],
        # contains
        ["contains", "Relation", df_contains.shape[0]],
        ["contains - high", "contains", 300],
        ["contains - low", "contains", 300],
        ["contains - pred", "contains", df_contains.shape[0] - 600],
        # isA
        ["isA", "Relation", df_isA.shape[0]],
        # hasChild
        ["hasChild", "Relation", df_hasChild.shape[0]],
        # hasPart
        ["hasPart", "Relation", df_hasPart.shape[0]],
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


def plot_sunburst_chemicals(fa_kg, G, dictionary):
    df_entities = fa_kg.get_all_entities()
    df_chemicals = df_entities[df_entities["type"].apply(lambda x: x == "chemical")]
    print(df_chemicals)

    df_evidence = fa_kg.get_evidence()
    not_in_list = ["NCBI_taxonomy", "MESH"]
    df_evidence = df_evidence[df_evidence["source"].apply(lambda x: x not in not_in_list)]
    entities_from_evidence = list(set(df_evidence["head"].tolist() + df_evidence["tail"].tolist()))
    print(df_evidence)
    print(set(df_evidence["source"]))

    rows = []
    count = 0
    for _, row in df_chemicals.iterrows():
        if "MESH_tree" not in row["other_db_ids"]:
            continue

        if row["foodatlas_id"] not in entities_from_evidence:
            continue

        count += 1

        for mesh_tree in row["other_db_ids"]["MESH_tree"]:
            mesh_tree_split = mesh_tree.split('.')

            if len(mesh_tree_split) <= 0:
                raise ValueError()

            new_row = []
            for idx in range(1, len(mesh_tree_split)+1):
                entity = fa_kg.get_entity_by_other_db_id("MESH_tree", '.'.join(mesh_tree_split[:idx]))
                new_row.append(entity["name"])
            rows.append(new_row)

    print(f"Processed {count} chemicals...")

    df = pd.DataFrame(rows)
    df.to_csv('temp.txt', sep='\t', index=False)
    df.replace(np.nan, '0', inplace=True)

    def _custom_wrap(s, width=20):
        return "<br>".join(textwrap.wrap(s, width=width))
    df = df.applymap(_custom_wrap)

    fig = px.sunburst(
        df,
        path=list(df.columns),
        maxdepth=2,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), font_family="Arial")
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_chemicals.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "sunburst_chemicals.svg"))


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
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(KG_DIR, "kg.txt"),
        evidence_filepath=os.path.join(KG_DIR, "evidence.txt"),
        entities_filepath=os.path.join(KG_DIR, "entities.txt"),
        relations_filepath=os.path.join(KG_DIR, "relations.txt"),
    )

    G, dictionary = generate_kg(fa_kg)

    # pie_chart_sources(fa_kg)
    # pie_chart_entities(fa_kg)
    # pie_chart_relations(fa_kg)
    # plot_sunburst_chemicals(fa_kg, G, dictionary)
    # chord_between_sources(fa_kg)
    plot_node_stats(fa_kg, G, dictionary)


if __name__ == "__main__":
    main()
