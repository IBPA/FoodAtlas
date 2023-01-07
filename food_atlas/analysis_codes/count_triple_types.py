import sys
sys.path.append('../data_processing/')
import pandas as pd  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

FINAL_DATA_DIR = "../../outputs/backend_data/v0.1"

fa_kg = KnowledgeGraph(kg_dir=FINAL_DATA_DIR)

df_evidence = fa_kg.get_evidence()
sources = set(df_evidence["source"].tolist())
print(f"Sources: {sources}")

df_annotation = df_evidence[df_evidence["source"]]
