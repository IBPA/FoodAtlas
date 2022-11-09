import sys
import pandas as pd

sys.path.append('./')
from .knowledge_graph import CandidateEntity, CandidateRelation  # noqa: E402


def read_tsv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t', keep_default_na=False)
    df["head"] = df["head"].apply(lambda x: eval(x, globals()))
    df["relation"] = df["relation"].apply(lambda x: eval(x, globals()))
    df["tail"] = df["tail"].apply(lambda x: eval(x, globals()))

    return df


def read_annotated(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t', keep_default_na=False)

    if "" in set(df["answer"].tolist()):
        raise ValueError("Make sure the 'answer' column does not contain empty response!")

    by = [
        "head",
        "relation",
        "tail",
        "pmid",
        "pmcid",
        "premise",
        "section",
        "hypothesis_id",
        "hypothesis_string",
        "source",
    ]
    df_grouped = df.groupby(by)["answer"].apply(list)
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped[df_grouped["answer"].apply(lambda x: len(set(x)) == 1)]
    df_grouped["answer"] = df_grouped["answer"].apply(lambda x: x[0])

    return df_grouped
