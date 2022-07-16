import argparse
from glob import glob
import os
from pathlib import Path
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.foodatlas_types import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.foodatlas_types import FoodAtlasEntity, FoodAtlasRelation  # noqa: E402


PH_PAIRS_FILEPATH = "../../output/data_generation/ph_pairs_*.txt"
PRE_ANNOTATION_FILEPATH = "../../output/data_generation/pre_annotation_*.tsv"
POST_ANNOTATION_FILEPATH = "../../output/data_generation/post_annotation_*.tsv"
NUM_ANNOTATE = 10
RANDOM_STATE = 530
KG_OUTPUT_DIR = "../../output/kg"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--ph_pairs_filepath",
        type=str,
        default=PH_PAIRS_FILEPATH,
        help=f"PH pairs filepath. (Default: {PH_PAIRS_FILEPATH})",
    )

    parser.add_argument(
        "--pre_annotation_filepath",
        type=str,
        default=PRE_ANNOTATION_FILEPATH,
        help=f"Annotation filepath. (Default: {PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--post_annotation_filepath",
        type=str,
        default=POST_ANNOTATION_FILEPATH,
        help=f"Annotation filepath. (Default: {POST_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--num_annotate",
        type=int,
        default=NUM_ANNOTATE,
        help=f"Number of PH pairs to annotate. (Default: {NUM_ANNOTATE})",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random state. (Default: {RANDOM_STATE})",
    )

    parser.add_argument(
        "--kg_output_dir",
        type=str,
        default=KG_OUTPUT_DIR,
        help="Directory to load/save the knowledge graph. "
             "By default, it uses last round to load "
             "and saves updates results to new folder."
             f"(Default: {KG_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    return args


def read_ph_pairs(ph_pairs_filepath: str) -> pd.DataFrame:
    if '*' in ph_pairs_filepath.split('/')[-1]:
        ph_pairs_filepath = sorted(glob(ph_pairs_filepath))[-1]
        print(f"Using the latest time-stamped PH pairs: {ph_pairs_filepath}")
    else:
        print(f"Using the user-specified PH pairs: {ph_pairs_filepath}")

    df = pd.read_csv(ph_pairs_filepath, sep='\t', keep_default_na=False)

    # def _read_foodatlasent(x):
    #     return eval(x, globals())
    # df["head"] = df["head"].apply(_read_foodatlasent)
    # df["relation"] = df["relation"].apply(_read_foodatlasent)
    # df["tail"] = df["tail"].apply(_read_foodatlasent)

    print(f"Input PH pairs dataframe shape: {df.shape}")

    return df


def find_what_to_do(
    pre_annotation_filepath: str,
    post_annotation_filepath: str,
) -> str:
    pre_files = sorted(glob(pre_annotation_filepath))
    post_files = sorted(glob(post_annotation_filepath))

    if len(pre_files) > 0:
        pre_rounds = [int(x.split('/')[-1].replace('.', '_').split('_')[-2])
                      for x in pre_files]
        assert set(pre_rounds) == set(range(1, max(pre_rounds)+1))
        max_pre_rounds = max(pre_rounds)
    else:
        max_pre_rounds = 0

    if len(post_files) > 0:
        post_rounds = [int(x.split('/')[-1].replace('.', '_').split('_')[-2])
                       for x in post_files]
        assert set(post_rounds) == set(range(1, max(post_rounds)+1))
        max_post_rounds = max(post_rounds)
    else:
        max_post_rounds = 0

    if max_pre_rounds == 0 and max_post_rounds == 0:
        todo = ["pre_annotation_1"]
    elif max_pre_rounds == max_post_rounds:
        todo = [f"post_process_annotation_{max_pre_rounds}", f"pre_annotation_{max_post_rounds+1}"]
    elif max_pre_rounds == (max_post_rounds + 1):
        raise RuntimeError(f"Need post annotation data: post_annotation_{max_pre_rounds}.tsv")
    else:
        raise ValueError("Invalid pre/post annotation sequence!")

    print(f"To do: {todo}")
    return todo


def generate_pre_annotation(
        df_ph_pairs: pd.DataFrame,
        pre_annotation_filepath: str,
        post_annotation_filepath: str,
        todo: str,
        num_annotate: int,
        random_state: int,
):
    cur_round = int(todo.replace("pre_annotation_", ""))
    print(f"Generate pre_annotation current round: {cur_round}")

    if cur_round == 1:
        df_to_annotate = df_ph_pairs.sample(n=num_annotate, random_state=random_state)
        df_to_annotate["round"] = cur_round
        filepath = pre_annotation_filepath.replace("*", str(cur_round))
        print(f"Saving {todo} to {filepath}")
        df_to_annotate.to_csv(filepath, sep='\t', index=False)

        return

    post_files = sorted(glob(post_annotation_filepath))
    data = []
    for f in post_files:
        data.append(pd.read_csv(f, sep='\t', keep_default_na=False))
    df_all_annotations = pd.concat(data)
    df_all_annotations = df_all_annotations[df_ph_pairs.columns]

    df_ph_pairs_without_annotated = pd.concat([df_ph_pairs, df_all_annotations])
    df_ph_pairs_without_annotated.drop_duplicates(keep=False, inplace=True)

    print(f"Original PH pairs dataframe shape: {df_ph_pairs.shape}")
    print(f"All previous annotations shape: {df_all_annotations.shape}")
    print(f"PH pairs without previous annotations shape: {df_ph_pairs_without_annotated.shape}")

    df_to_annotate = df_ph_pairs_without_annotated.sample(n=num_annotate, random_state=random_state)
    df_to_annotate["round"] = cur_round
    filepath = pre_annotation_filepath.replace("*", str(cur_round))
    print(f"Saving {todo} to {filepath}")
    df_to_annotate.to_csv(filepath, sep='\t', index=False)


def post_process_annotation(
    pre_annotation_filepath: str,
    post_annotation_filepath: str,
    kg_output_dir: str,
    todo: str,
    num_annotate: int,
    random_state: int,
    kg_filename: str = "kg.txt",
    entities_filename: str = "entities.txt",
    relations_filename: str = "relations.txt",
):
    post_files = sorted(glob(post_annotation_filepath))
    this_round = int(post_files[-1].split('/')[-1].replace('.', '_').split('_')[-2])
    print(f"Post processing round: {this_round}")

    # parse annotation to KG
    if this_round == 1:
        new_kg_folder = os.path.join(kg_output_dir, str(this_round))
        Path(new_kg_folder).mkdir(parents=True, exist_ok=False)
        fa_ent = FoodAtlasEntity(os.path.join(new_kg_folder, entities_filename))
        fa_rel = FoodAtlasRelation(os.path.join(new_kg_folder, relations_filename))
    else:
        old_kg_folder = os.path.join(kg_output_dir, str(this_round-1))
        df_kg_old = pd.read_csv(os.path.join(old_kg_folder, kg_filename), sep='\t', keep_default_na=False)
        fa_ent = FoodAtlasEntity(os.path.join(old_kg_folder, entities_filename))
        fa_rel = FoodAtlasRelation(os.path.join(old_kg_folder, relations_filename))

        new_kg_folder = os.path.join(kg_output_dir, str(this_round))
        # Path(new_kg_folder).mkdir(parents=True, exist_ok=False)

    def _read_foodatlasent(x):
        return eval(x, globals())

    df_last_annotation = pd.read_csv(post_files[-1], sep='\t', keep_default_na=False)
    df_last_annotation["head"] = df_last_annotation["head"].apply(_read_foodatlasent)
    df_last_annotation["relation"] = df_last_annotation["relation"].apply(_read_foodatlasent)
    df_last_annotation["tail"] = df_last_annotation["tail"].apply(_read_foodatlasent)
    pd_last_pos = df_last_annotation[df_last_annotation["answer"] == "Entails"]

    data = []
    for idx, row in pd_last_pos.iterrows():
        head = fa_ent.add(
            type=row["head"].type,
            name=row["head"].name,
            synonyms=row["head"].synonyms,
            other_db_ids=row["head"].other_db_ids,
        )

        relation = fa_rel.add(
            name=row["relation"].name,
            translation=row["relation"].translation,
        )

        tail = fa_ent.add(
            type=row["tail"].type,
            name=row["tail"].name,
            synonyms=row["tail"].synonyms,
            other_db_ids=row["tail"].other_db_ids,
        )

        source = {
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "section": row["section"],
            "premise": row["premise"],
        }

        annotation_info = {
            "round": row["round"],
            "id": row["id"],
            "annotator": row["annotator"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

        newrow = pd.Series({
            "head": head.foodatlas_id,
            "relation": relation.foodatlas_id,
            "tail": tail.foodatlas_id,
            "sources": [source],
            "annotation_infos": [annotation_info],
        })

        data.append(newrow)

        # has part
        if row["head"].type == "species_with_part":
            head = fa_ent.add(
                type="species",
                name=row["head"].name.split(" - ")[0],
            )

            relation = fa_rel.add(
                name="hasPart",
                translation="has part",
            )

            tail = fa_ent.add(
                type=row["head"].type,
                name=row["head"].name,
                synonyms=row["head"].synonyms,
                other_db_ids=row["head"].other_db_ids,
            )

            newrow = pd.Series({
                "head": head.foodatlas_id,
                "relation": relation.foodatlas_id,
                "tail": tail.foodatlas_id,
                "sources": [source],
                "annotation_infos": [annotation_info],
            })

            data.append(newrow)

    df_kg_new = pd.DataFrame(data).reset_index(drop=True)

    if this_round == 1:
        df_kg_new.to_csv(os.path.join(new_kg_folder, kg_filename), sep='\t', index=False)
        fa_ent.save()
        fa_rel.save()
    else:
        df_kg_new = pd.concat([df_kg_old, df_kg_new])
        df_kg_new.to_csv(os.path.join(new_kg_folder, kg_filename), sep='\t', index=False)
        fa_ent.save(os.path.join(new_kg_folder, entities_filename))
        fa_rel.save(os.path.join(new_kg_folder, relations_filename))


def main():
    args = parse_argument()

    df_ph_pairs = read_ph_pairs(args.ph_pairs_filepath)

    todo = find_what_to_do(
        pre_annotation_filepath=args.pre_annotation_filepath,
        post_annotation_filepath=args.post_annotation_filepath,
    )

    if len(todo) == 1:
        generate_pre_annotation(
            df_ph_pairs=df_ph_pairs,
            pre_annotation_filepath=args.pre_annotation_filepath,
            post_annotation_filepath=args.post_annotation_filepath,
            todo=todo[0],
            num_annotate=args.num_annotate,
            random_state=args.random_state,
        )
    else:
        post_process_annotation(
            pre_annotation_filepath=args.pre_annotation_filepath,
            post_annotation_filepath=args.post_annotation_filepath,
            kg_output_dir=args.kg_output_dir,
            todo=todo[0],
            num_annotate=args.num_annotate,
            random_state=args.random_state,
        )

        generate_pre_annotation(
            df_ph_pairs=df_ph_pairs,
            pre_annotation_filepath=args.pre_annotation_filepath,
            post_annotation_filepath=args.post_annotation_filepath,
            todo=todo[1],
            num_annotate=args.num_annotate,
            random_state=args.random_state,
        )


if __name__ == '__main__':
    main()
