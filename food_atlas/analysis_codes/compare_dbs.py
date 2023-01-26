import os
import sys
import textwrap
import random

sys.path.append('../data_processing/')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from matplotlib_venn import venn3  # noqa: E402

FINAL_DATA_DIR = "../../outputs/backend_data/v0.1"
OUTPUT_DIR = "../../outputs/analysis_codes"

fa_kg = KnowledgeGraph(kg_dir=FINAL_DATA_DIR)
df_evidence = fa_kg.get_evidence()
sources = list(set(df_evidence['source'].tolist()))
print(sources)

df_grouped = df_evidence.groupby("source")["triple"].apply(set).reset_index()
st_dict = dict(zip(df_grouped["source"].tolist(), df_grouped["triple"].tolist()))

frida_annotation = \
    st_dict["Frida"] & \
    st_dict["FoodAtlas:annotation"] - \
    st_dict["FoodAtlas:prediction:entailment"] - \
    st_dict["NCBI_taxonomy"] - \
    st_dict["Phenol-Explorer"] - \
    st_dict["MeSH"]

frida = \
    st_dict["Frida"] - \
    st_dict["FoodAtlas:annotation"] - \
    st_dict["FoodAtlas:prediction:entailment"] - \
    st_dict["NCBI_taxonomy"] - \
    st_dict["Phenol-Explorer"] - \
    st_dict["MeSH"]

annotation = \
    st_dict["FoodAtlas:annotation"] - \
    st_dict["Frida"] - \
    st_dict["FoodAtlas:prediction:entailment"] - \
    st_dict["NCBI_taxonomy"] - \
    st_dict["Phenol-Explorer"] - \
    st_dict["MeSH"]

print(frida_annotation)

# food?
frida_foods = set(df_evidence[df_evidence["source"] == "Frida"]["head"].tolist())
phenol_explorer_foods = set(df_evidence[df_evidence["source"] == "Phenol-Explorer"]["head"].tolist())
foodatlas_foods = set(df_evidence[df_evidence["source"].apply(lambda x: x.startswith("FoodAtlas:"))]["head"].tolist())
print(f"len(frida_foods): {len(frida_foods)}")
print(f"len(phenol_explorer_foods): {len(phenol_explorer_foods)}")
print(f"len(foodatlas_foods): {len(foodatlas_foods)}")

# chemical?
frida_chemicals = set(df_evidence[df_evidence["source"] == "Frida"]["tail"].tolist())
phenol_explorer_chemicals = set(df_evidence[df_evidence["source"] == "Phenol-Explorer"]["tail"].tolist())
foodatlas_chemicals = set(df_evidence[df_evidence["source"].apply(lambda x: x.startswith("FoodAtlas:"))]["tail"].tolist())
print(f"len(frida_chemicals): {len(frida_chemicals)}")
print(f"len(phenol_explorer_chemicals): {len(phenol_explorer_chemicals)}")
print(f"len(foodatlas_chemicals): {len(foodatlas_chemicals)}")

plt.figure()
venn3(
    [frida_foods, phenol_explorer_foods, foodatlas_foods],
    ('Frida', 'Phenol-Explorer', 'FoodAtlas')
)
plt.savefig(os.path.join(OUTPUT_DIR, "food_venn_diagram.png"))

plt.figure()
venn3(
    [frida_chemicals, phenol_explorer_chemicals, foodatlas_chemicals],
    ('Frida', 'Phenol-Explorer', 'FoodAtlas')
)
plt.savefig(os.path.join(OUTPUT_DIR, "chemical_venn_diagram.png"))






# CHANGE IT SO THAT FOOD AND ORGANISMS ARE NOT ONTOLOGICAL/TAXONOMICAL


