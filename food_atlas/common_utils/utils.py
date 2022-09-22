import ast

import pandas as pd
from tqdm import tqdm

from common_utils.foodatlas_types import CandidateEntity, CandidateRelation


def read_kg(filepath) -> pd.DataFrame:
    df_kg = pd.read_csv(filepath, sep='\t', keep_default_na=False)
    df_kg["evidence"] = df_kg["evidence"].apply(ast.literal_eval)

    return df_kg


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


def generate_kg(
        df_input,
        fa_ent,
        fa_rel,
        existing_kg=None,
        source: str = None,
) -> pd.DataFrame:
    data = []
    for idx, row in tqdm(df_input.iterrows(), total=df_input.shape[0]):
        head = row.at["head"]
        tail = row.at["tail"]
        hypothesis_id = row.at["hypothesis_id"]

        # if source == "annotation":
        #     round_ = row.at["round"]
        #     source = f"{source}:round_{round_}"
        # elif source == "prediction":
        #     prob = row.at["prob"]
        #     source = f"{source}:prob_{prob}"

        # head
        if head.type == "species_with_part":
            other_db_ids = dict([hypothesis_id.split('-')[0].split(':')])
        else:
            other_db_ids = head.other_db_ids

        head_ent = fa_ent.add(
            type_=head.type,
            name=head.name,
            synonyms=head.synonyms,
            other_db_ids=other_db_ids,
        )

        # relation
        relation = fa_rel.add(
            name=row["relation"].name,
            translation=row["relation"].translation,
        )

        # tail
        if tail.type == "species_with_part":
            other_db_ids = dict([hypothesis_id.split('-')[0].split(':')])
        else:
            other_db_ids = tail.other_db_ids

        tail_ent = fa_ent.add(
            type_=tail.type,
            name=tail.name,
            synonyms=tail.synonyms,
            other_db_ids=other_db_ids,
        )

        newrow = pd.Series({
            "head": head_ent.foodatlas_id,
            "relation": relation.foodatlas_id,
            "tail": tail_ent.foodatlas_id,
            "evidence": [{
                "pmid": row["pmid"],
                "pmcid": row["pmcid"],
                "section": row["section"],
                "premise": row["premise"],
                "round": row["round"],
            }],
            # "source":,
            # "probability",
        })

        data.append(newrow)

        # has part
        if head.type == "species_with_part":
            tail_ent = head_ent
            head_ent = fa_ent.add(
                type_="species",
                name=head.name.split(" - ")[0],
                other_db_ids=dict([hypothesis_id.split('-')[0].split(':')]),
            )

            relation = fa_rel.add(
                name="hasPart",
                translation="has part",
            )

            newrow = pd.Series({
                "head": head_ent.foodatlas_id,
                "relation": relation.foodatlas_id,
                "tail": tail_ent.foodatlas_id,
                "evidence": [{
                    "pmid": row["pmid"],
                    "pmcid": row["pmcid"],
                    "section": row["section"],
                    "premise": row["premise"],
                    "round": row["round"],
                }],
                # "source":,
                # "probability",
            })

            data.append(newrow)

    df_kg = pd.DataFrame(data)
    df_kg = df_kg.groupby(["head", "relation", "tail"])["evidence"].agg(sum).reset_index()

    def _merge_evidence(x):
        return [dict(t) for t in {tuple(d.items()) for d in x}]
    df_kg["evidence"] = df_kg["evidence"].apply(_merge_evidence)
    df_kg["num_evidence"] = df_kg["evidence"].apply(len)

    return df_kg
