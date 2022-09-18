import argparse
from glob import glob
import os
from pathlib import Path
import random
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.foodatlas_types import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.foodatlas_types import FoodAtlasEntity, FoodAtlasRelation  # noqa: E402


PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_*.txt"
PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/pre_annotation_*.tsv"
POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/post_annotation_*.tsv"
RANDOM_STATE = 530
KG_OUTPUT_DIR = "../../outputs/kg"
REMAINING_FILEPATH = "../../outputs/data_generation/remaining.tsv"
VAL_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_pre_annotation.tsv"
TEST_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_pre_annotation.tsv"
VAL_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_post_annotation.tsv"
TEST_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_post_annotation.tsv"
TRAIN_FILEPATH = "../../outputs/data_generation/train.tsv"
VAL_FILEPATH = "../../outputs/data_generation/val.tsv"
TEST_FILEPATH = "../../outputs/data_generation/test.tsv"
TO_PREDICT_FILEPATH = "../../outputs/data_generation/to_predict_*.tsv"
VAL_NUM_PREMISE = 500
TEST_NUM_PREMISE = 500
PER_ROUND_NUM_PREMISE = 500
NUM_AUGMENT = 10


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

    parser.add_argument(
        "--remaining_filepath",
        type=str,
        default=REMAINING_FILEPATH,
        help=f"Remaining PH pairs filepath. (Default: {REMAINING_FILEPATH})",
    )

    parser.add_argument(
        "--val_num_premise",
        type=int,
        default=VAL_NUM_PREMISE,
        help=f"Number of premises to use for validation. (Default: {VAL_NUM_PREMISE})",
    )

    parser.add_argument(
        "--val_pre_annotation_filepath",
        type=str,
        default=VAL_PRE_ANNOTATION_FILEPATH,
        help=f"Validation pre-annotation set filepath. (Default: {VAL_PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--test_num_premise",
        type=int,
        default=TEST_NUM_PREMISE,
        help=f"Number of premises to use for test. (Default: {TEST_NUM_PREMISE})",
    )

    parser.add_argument(
        "--per_round_num_premise",
        type=int,
        default=PER_ROUND_NUM_PREMISE,
        help=f"Number of premises to use per round. (Default: {PER_ROUND_NUM_PREMISE})",
    )

    parser.add_argument(
        "--test_pre_annotation_filepath",
        type=str,
        default=TEST_PRE_ANNOTATION_FILEPATH,
        help=f"Test set pre-annotation filepath. (Default: {TEST_PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--val_post_annotation_filepath",
        type=str,
        default=VAL_POST_ANNOTATION_FILEPATH,
        help=f"Validation set post-annotation filepath. (Default: {VAL_POST_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--test_post_annotation_filepath",
        type=str,
        default=TEST_POST_ANNOTATION_FILEPATH,
        help=f"Test set post-annotation filepath. (Default: {TEST_POST_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--train_filepath",
        type=str,
        default=TRAIN_FILEPATH,
        help=f"Train filepath. (Default: {TRAIN_FILEPATH})",
    )

    parser.add_argument(
        "--val_filepath",
        type=str,
        default=VAL_FILEPATH,
        help=f"Validation set final processed filepath. (Default: {VAL_FILEPATH})",
    )

    parser.add_argument(
        "--test_filepath",
        type=str,
        default=TEST_FILEPATH,
        help=f"Test set final processed filepath. (Default: {TEST_FILEPATH})",
    )

    parser.add_argument(
        "--to_predict_filepath",
        type=str,
        default=TO_PREDICT_FILEPATH,
        help=f"To predict filepath. (Default: {TO_PREDICT_FILEPATH})",
    )

    parser.add_argument(
        "--skip_augment",
        action="store_true",
        help="Set if skipping augmentation.",
    )

    parser.add_argument(
        "--augment_num",
        type=int,
        default=NUM_AUGMENT,
        help=f"How many augmentations per triple. (Default: {NUM_AUGMENT})",
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
        per_round_num_premise: int,
        random_state: int,
        remaining_filepath: str,
        to_predict_filepath: str,
        val_num_premise: int = None,
        val_pre_annotation_filepath: str = None,
        test_num_premise: int = None,
        test_pre_annotation_filepath: str = None,
):
    cur_round = int(todo.replace("pre_annotation_", ""))
    print(f"Generate pre_annotation current round: {cur_round}")

    if cur_round == 1:
        premises = list(set(df_ph_pairs["premise"].tolist()))
        random.shuffle(premises)
        val_premises = premises[0:val_num_premise]
        test_premises = premises[val_num_premise:val_num_premise+test_num_premise]
        remaining_premises = premises[val_num_premise+test_num_premise:]

        df_remaining = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in remaining_premises)]
        df_val = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in val_premises)]
        df_test = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in test_premises)]

        print(f"df_ph_pairs shape: {df_ph_pairs.shape}")
        print(f"df_remaining shape: {df_remaining.shape}")
        print(f"df_val shape: {df_val.shape}")
        print(f"df_test shape: {df_test.shape}")

        df_remaining.to_csv(remaining_filepath, sep='\t', index=False)
        df_val.to_csv(val_pre_annotation_filepath, sep='\t', index=False)
        df_test.to_csv(test_pre_annotation_filepath, sep='\t', index=False)

        this_round_premises = random.sample(remaining_premises, per_round_num_premise)
        df_to_annotate = df_remaining[df_remaining["premise"].apply(
            lambda x: x in this_round_premises)]
        df_to_annotate["round"] = cur_round
        filepath = pre_annotation_filepath.replace("*", str(cur_round))
        print(f"Saving {todo} to {filepath}")
        df_to_annotate.to_csv(filepath, sep='\t', index=False)

        df_to_test = df_remaining[df_remaining["premise"].apply(
            lambda x: x not in this_round_premises)]
        filepath = to_predict_filepath.replace("*", str(cur_round))
        print(f"Saving to_predict to {filepath}")
        df_to_test.to_csv(filepath, sep='\t', index=False)

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

    #  modify!!
    remaining_premises = list(set(df_ph_pairs_without_annotated["premise"].tolist()))
    this_round_premises = random.sample(remaining_premises, per_round_num_premise)
    df_to_annotate = df_ph_pairs_without_annotated[df_ph_pairs_without_annotated["premise"].apply(
        lambda x: x in this_round_premises)]
    df_to_annotate["round"] = cur_round
    filepath = pre_annotation_filepath.replace("*", str(cur_round))
    print(f"Saving {todo} to {filepath}")
    df_to_annotate.to_csv(filepath, sep='\t', index=False)


def post_process_annotation(
    post_annotation_filepath: str,
    kg_output_dir: str,
    todo: str,
    random_state: int,
    kg_filename: str = "kg.txt",
    entities_filename: str = "entities.txt",
    relations_filename: str = "relations.txt",
    val_post_annotation_filepath: str = None,
    test_post_annotation_filepath: str = None,
    train_filepath: str = None,
    val_filepath: str = None,
    test_filepath: str = None,
    skip_augment: bool = False,
    augment_num: int = None,
):
    post_files = sorted(glob(post_annotation_filepath))
    this_round = int(post_files[-1].split('/')[-1].replace('.', '_').split('_')[-2])
    print(f"Post processing round: {this_round}")

    def _read_annotated(filename):
        df = pd.read_csv(filename, sep='\t', keep_default_na=False)
        df["head"] = df["head"].apply(lambda x: eval(x, globals()))
        df["relation"] = df["relation"].apply(lambda x: eval(x, globals()))
        df["tail"] = df["tail"].apply(lambda x: eval(x, globals()))

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

    # parse annotation to KG
    if this_round == 1:
        new_kg_folder = os.path.join(kg_output_dir, str(this_round))
        Path(new_kg_folder).mkdir(parents=True, exist_ok=True)
        fa_ent = FoodAtlasEntity(os.path.join(new_kg_folder, entities_filename))
        fa_rel = FoodAtlasRelation(os.path.join(new_kg_folder, relations_filename))

        # val and test need to be cleaned up
        df_val = _read_annotated(val_post_annotation_filepath)
        df_test = _read_annotated(test_post_annotation_filepath)

        df_val.to_csv(val_filepath, sep='\t', index=False)
        df_test.to_csv(test_filepath, sep='\t', index=False)
    else:
        old_kg_folder = os.path.join(kg_output_dir, str(this_round-1))
        df_kg_old = pd.read_csv(os.path.join(old_kg_folder, kg_filename), sep='\t', keep_default_na=False)
        fa_ent = FoodAtlasEntity(os.path.join(old_kg_folder, entities_filename))
        fa_rel = FoodAtlasRelation(os.path.join(old_kg_folder, relations_filename))

        # new_kg_folder = os.path.join(kg_output_dir, str(this_round))
        # Path(new_kg_folder).mkdir(parents=True, exist_ok=False)

    df_last_annotation = _read_annotated(post_files[-1])

    # generate KG
    print("Generating KG...")

    if this_round == 1:
        df_val["round"] = "val"
        df_test["round"] = "test"

        df_val = df_val[df_last_annotation.columns]
        df_test = df_test[df_last_annotation.columns]

        df_for_kg = pd.concat([df_val, df_test, df_last_annotation]).reset_index(drop=True)
    else:
        df_for_kg = df_last_annotation

    df_pos = df_for_kg[df_for_kg["answer"] == "Entails"]
    df_pos.reset_index(inplace=True, drop=True)

    data = []
    for idx, row in df_pos.iterrows():
        head = row.at["head"]
        tail = row.at["tail"]
        hypothesis_id = row.at["hypothesis_id"]

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
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "section": row["section"],
            "premise": row["premise"],
            "round": row["round"],
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
                "pmid": row["pmid"],
                "pmcid": row["pmcid"],
                "section": row["section"],
                "premise": row["premise"],
                "round": row["round"],
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

    # generate training data
    print("Generating training data...")

    df_train = df_last_annotation.copy()[
        ["head", "relation", "tail", "premise", "hypothesis_string", "hypothesis_id", "answer"]
    ]

    if skip_augment:
        df_train["augmentation"] = "original"
    else:
        df_species = fa_ent.get_entities_by_type(type_="species")
        df_chemical = fa_ent.get_entities_by_type(type_="chemical")

        augmented = []
        for idx, row in df_train.iterrows():
            orig = row.to_dict()
            orig["augmentation"] = "original"
            augmented.append(orig)

            head = row.at["head"]
            tail = row.at["tail"]
            hypothesis = row.at["hypothesis_string"].lower()
            premise = row.at["premise"].lower()

            if random.random() < 0.5:  # head
                for _ in range(augment_num):
                    aug = row.to_dict()
                    aug["augmentation"] = "replace_head"

                    replace_from_candidates = [head.name] + head.synonyms
                    replace_from_candidates = [x.lower() for x in replace_from_candidates]
                    if head.type == "species_with_part":
                        replace_from_candidates = [x.split(" - ")[0] for x in replace_from_candidates]
                    replace_from_candidates.sort(key=len, reverse=True)
                    replace_to = replace_from_candidates.copy()

                    if "species" in head.type:
                        df_sampling_pool = df_species
                    else:
                        raise ValueError()

                    while set(replace_to) == set(replace_from_candidates):
                        replace_to = df_sampling_pool.sample(n=1).iloc[0]
                        replace_to = [replace_to.name] + replace_to.synonyms
                        replace_to = [x.lower() for x in replace_to]
                    replace_to = random.choice(replace_to)

                    replace_from = None
                    for x in replace_from_candidates:
                        if x in premise and x in hypothesis:
                            replace_from = x
                            break

                    if replace_from is None:
                        print(f"premise: {premise}")
                        print(f"hypothesis: {hypothesis}")
                        print(f"replace_from_candidates: {replace_from_candidates}")
                        raise ValueError("Could not find entity in the premise and hypothesis!")

                    aug["premise"] = premise.replace(replace_from, replace_to)
                    aug["hypothesis_string"] = hypothesis.replace(replace_from, replace_to)

                    augmented.append(aug)
            else:  # tail
                for _ in range(augment_num):
                    aug = row.to_dict()
                    aug["augmentation"] = "replace_tail"

                    replace_from_candidates = [tail.name] + tail.synonyms
                    replace_from_candidates = [x.lower() for x in replace_from_candidates]
                    if tail.type == "species_with_part":
                        replace_from_candidates = [x.split(" - ")[0] for x in replace_from_candidates]
                    replace_from_candidates.sort(key=len, reverse=True)
                    replace_to = replace_from_candidates.copy()

                    if "chemical" in tail.type:
                        df_sampling_pool = df_chemical
                    else:
                        raise ValueError()

                    while set(replace_to) == set(replace_from_candidates):
                        replace_to = df_sampling_pool.sample(n=1).iloc[0]
                        replace_to = [replace_to.name] + replace_to.synonyms
                        replace_to = [x.lower() for x in replace_to]
                    replace_to = random.choice(replace_to)

                    replace_from = None
                    for x in replace_from_candidates:
                        if x in premise and x in hypothesis:
                            replace_from = x
                            break

                    if replace_from is None:
                        print(f"premise: {premise}")
                        print(f"hypothesis: {hypothesis}")
                        print(f"replace_from_candidates: {replace_from_candidates}")
                        raise ValueError("Could not find entity in the premise and hypothesis!")

                    aug["premise"] = premise.replace(replace_from, replace_to)
                    aug["hypothesis_string"] = hypothesis.replace(replace_from, replace_to)

                    augmented.append(aug)

        df_train = pd.DataFrame(augmented)

    df_train = df_train[["premise", "hypothesis_string", "hypothesis_id", "answer", "augmentation"]]
    df_train.to_csv(
        train_filepath.replace('.tsv', f'_{this_round}.tsv'),
        sep='\t',
        index=False
    )

    sys.exit()


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
            per_round_num_premise=args.per_round_num_premise,
            random_state=args.random_state,
            remaining_filepath=args.remaining_filepath,
            to_predict_filepath=args.to_predict_filepath,
            val_num_premise=args.val_num_premise,
            val_pre_annotation_filepath=args.val_pre_annotation_filepath,
            test_num_premise=args.test_num_premise,
            test_pre_annotation_filepath=args.test_pre_annotation_filepath,
        )
    else:
        post_process_annotation(
            post_annotation_filepath=args.post_annotation_filepath,
            kg_output_dir=args.kg_output_dir,
            todo=todo[0],
            random_state=args.random_state,
            val_post_annotation_filepath=args.val_post_annotation_filepath,
            test_post_annotation_filepath=args.test_post_annotation_filepath,
            train_filepath=args.train_filepath,
            val_filepath=args.val_filepath,
            test_filepath=args.test_filepath,
            skip_augment=args.skip_augment,
            augment_num=args.augment_num,
        )

        generate_pre_annotation(
            df_ph_pairs=df_ph_pairs,
            pre_annotation_filepath=args.pre_annotation_filepath,
            post_annotation_filepath=args.post_annotation_filepath,
            todo=todo[1],
            per_round_num_premise=args.per_round_num_premise,
            random_state=args.random_state,
            remaining_filepath=args.remaining_filepath,
            to_predict_filepath=args.to_predict_filepath,
        )


if __name__ == '__main__':
    main()
