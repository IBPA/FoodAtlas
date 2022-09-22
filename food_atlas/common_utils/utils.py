import sys
import pandas as pd

sys.path.append('./')
from .knowledge_graph import KnowledgeGraph  # noqa: E402
CandidateEntity = KnowledgeGraph.CandidateEntity
CandidateRelation = KnowledgeGraph.CandidateRelation


def read_tsv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t', keep_default_na=False)
    df["head"] = df["head"].apply(lambda x: eval(x, globals()))
    df["relation"] = df["relation"].apply(lambda x: eval(x, globals()))
    df["tail"] = df["tail"].apply(lambda x: eval(x, globals()))

    return df


def read_annotated(filepath) -> pd.DataFrame:
    df = read_tsv(filepath)

    if "" in set(df["answer"].tolist()):
        raise ValueError("Make sure the 'answer' column does not contain empty response!")

    answer_agrees = df.groupby('id')['answer'].apply(lambda x: len(set(x)) == 1)
    agreement_ids = answer_agrees[answer_agrees].index.tolist()

    df = df[df["id"].apply(lambda x: x in agreement_ids)]
    df.drop_duplicates("id", inplace=True, ignore_index=True)
    df.drop(
        ["id", "annotator", "annotation_id", "created_at", "updated_at", "lead_time"],
        inplace=True,
        axis=1)

    return df
