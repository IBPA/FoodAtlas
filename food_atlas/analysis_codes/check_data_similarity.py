import os
import sys
sys.path.append('..')
from tqdm import trange  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.express as px  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from data_processing.common_utils.utils import load_pkl, save_pkl  # noqa: E402
from scipy.stats import ttest_ind

ROOT_DIR = "/home/jasonyoun/Temp/data_generation"
OUTPUT_DIR = "../../outputs/analysis_codes"

AL_STRATEGIES = [
    "certain_pos",
    "stratified",
    "uncertain",
    "random",
]

# pos_dict = {al: {i+1: [] for i in range(10)} for al in AL_STRATEGIES}
# neg_dict = {al: {i+1: [] for i in range(10)} for al in AL_STRATEGIES}

# for al in AL_STRATEGIES:
#     print(f"Processing: {al}...")
#     for run in trange(1, 101):
#         for round in range(1, 11):
#             filepath = os.path.join(ROOT_DIR, al, f"run_{run}", f"round_{round}", "train.tsv")
#             df = pd.read_csv(filepath, sep='\t', keep_default_na=False)
#             answers = df["answer"].tolist()
#             num_pos = answers.count("Entails")
#             num_neg = answers.count("Does not entail")
#             pos_dict[al][round].append(num_pos)
#             neg_dict[al][round].append(num_neg)

# save_pkl(pos_dict, os.path.join(OUTPUT_DIR, 'num_pos_dict.pkl'))
# save_pkl(neg_dict, os.path.join(OUTPUT_DIR, 'num_neg_dict.pkl'))

pos_dict = load_pkl(os.path.join(OUTPUT_DIR, 'num_pos_dict.pkl'))
neg_dict = load_pkl(os.path.join(OUTPUT_DIR, 'num_neg_dict.pkl'))

certain_pos_ratio = {r: [] for r in range(1, 11)}
stratified_ratio = {r: [] for r in range(1, 11)}
uncertain_ratio = {r: [] for r in range(1, 11)}
random_ratio = {r: [] for r in range(1, 11)}
for round in range(1, 11):
    for run in range(100):
        num_pos = pos_dict["certain_pos"][round][run]
        num_neg = neg_dict["certain_pos"][round][run]
        certain_pos_ratio[round].append(num_pos / (num_pos + num_neg))

        num_pos = pos_dict["stratified"][round][run]
        num_neg = neg_dict["stratified"][round][run]
        stratified_ratio[round].append(num_pos / (num_pos + num_neg))

        num_pos = pos_dict["uncertain"][round][run]
        num_neg = neg_dict["uncertain"][round][run]
        uncertain_ratio[round].append(num_pos / (num_pos + num_neg))

        num_pos = pos_dict["random"][round][run]
        num_neg = neg_dict["random"][round][run]
        random_ratio[round].append(num_pos / (num_pos + num_neg))

rows = []
for round in range(1, 11):
    rows.append(["val", round, 295/825, 0])
    rows.append(["test", round, 312/840, 0])

    rows.append([
        "certain_pos",
        round,
        np.mean(certain_pos_ratio[round]),
        np.std(certain_pos_ratio[round]),
    ])

    rows.append([
        "stratified",
        round,
        np.mean(stratified_ratio[round]),
        np.std(stratified_ratio[round]),
    ])

    rows.append([
        "uncertain",
        round,
        np.mean(uncertain_ratio[round]),
        np.std(uncertain_ratio[round]),
    ])

    rows.append([
        "random",
        round,
        np.mean(random_ratio[round]),
        np.std(random_ratio[round]),
    ])

df_plot = pd.DataFrame(rows, columns=["strategy", "round", "percentage of positives", "std"])
print(df_plot)

fig = px.line(
    df_plot,
    x="round",
    y="percentage of positives",
    error_y="std",
    color="strategy",
    markers=True,
    width=800,
    height=600,
)
fig.update_layout(xaxis=dict(tickmode='linear'))
fig.write_image(os.path.join(OUTPUT_DIR, "data_balance.png"))
fig.write_image(os.path.join(OUTPUT_DIR, "data_balance.svg"))

# p-values
for round in range(1, 11):
    print(f"Round: {round}")

    r = ttest_ind(certain_pos_ratio[round], random_ratio[round])
    print(f"certain_pos vs. random: {r[1]}")

    r = ttest_ind(stratified_ratio[round], random_ratio[round])
    print(f"stratified_ratio vs. random: {r[1]}")

    r = ttest_ind(uncertain_ratio[round], random_ratio[round])
    print(f"uncertain_ratio vs. random: {r[1]}")

    r = ttest_ind(certain_pos_ratio[round], stratified_ratio[round])
    print(f"certain_pos vs. stratified_ratio: {r[1]}")

    r = ttest_ind(certain_pos_ratio[round], uncertain_ratio[round])
    print(f"certain_pos vs. uncertain_ratio: {r[1]}")

    r = ttest_ind(stratified_ratio[round], uncertain_ratio[round])
    print(f"stratified_ratio vs. uncertain_ratio: {r[1]}")

    print()
