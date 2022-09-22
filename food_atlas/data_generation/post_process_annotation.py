import argparse
import os
from pathlib import Path
import random
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.utils import read_annotated  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph


POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/post_annotation_*.tsv"
RANDOM_STATE = 530
KG_OUTPUT_DIR = "../../outputs/kg"
VAL_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_post_annotation.tsv"
TEST_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_post_annotation.tsv"
TRAIN_FILEPATH = "../../outputs/data_generation/train.tsv"
VAL_FILEPATH = "../../outputs/data_generation/val.tsv"
TEST_FILEPATH = "../../outputs/data_generation/test.tsv"
KG_FILENAME = "kg.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
NUM_AUGMENT = 10


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="What pre_annotation round are we generating.",
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
        help="Set random state.",
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


def export_to_kg(df_annotated, args):
    # generate KG
    print("Generating KG...")

    if args.round == 1:
        new_kg_folder = os.path.join(args.kg_output_dir, str(args.round))
        Path(new_kg_folder).mkdir(parents=True, exist_ok=True)

        # val and test need to be cleaned up
        df_val = read_annotated(args.val_post_annotation_filepath)
        df_test = read_annotated(args.test_post_annotation_filepath)
        df_val.to_csv(args.val_filepath, sep='\t', index=False)
        df_test.to_csv(args.test_filepath, sep='\t', index=False)

        df_val["round"] = "val"
        df_test["round"] = "test"

        df_val = df_val[df_annotated.columns]
        df_test = df_test[df_annotated.columns]

        df_for_kg = pd.concat([df_val, df_test, df_annotated]).reset_index(drop=True)
    else:
        df_for_kg = df_annotated
        raise NotImplementedError()

    df_pos = df_for_kg[df_for_kg["answer"] == "Entails"]
    df_pos.reset_index(inplace=True, drop=True)

    df_pos = df_pos.head(100)

    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(new_kg_folder, KG_FILENAME),
        entities_filepath=os.path.join(new_kg_folder, ENTITIES_FILENAME),
        relations_filepath=os.path.join(new_kg_folder, RELATIONS_FILENAME),
    )
    fa_kg.add_ph_pairs(df_pos)

    if args.round == 1:
        fa_kg.save()
    else:
        raise NotImplementedError()

    return fa_kg


def generate_training(df_annotated, fa_kg, args):
    df_train = df_annotated.copy()[
        ["head", "relation", "tail", "premise", "hypothesis_string", "hypothesis_id", "answer"]
    ]

    if args.skip_augment:
        df_train["augmentation"] = "original"
    else:
        df_species = fa_kg.get_entities_by_type(type_="species")
        df_chemical = fa_kg.get_entities_by_type(type_="chemical")

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
                for _ in range(args.augment_num):
                    aug = row.to_dict()
                    aug["augmentation"] = "replace_head"

                    replace_from_cand = [head.name] + head.synonyms
                    replace_from_cand = [x.lower() for x in replace_from_cand]
                    if head.type == "species_with_part":
                        replace_from_cand = [x.split(" - ")[0] for x in replace_from_cand]
                    replace_from_cand.sort(key=len, reverse=True)
                    replace_to = replace_from_cand.copy()

                    if "species" in head.type:
                        df_sampling_pool = df_species
                    else:
                        raise ValueError()

                    while set(replace_to) == set(replace_from_cand):
                        replace_to = df_sampling_pool.sample(n=1).iloc[0]
                        replace_to = [replace_to.name] + replace_to.synonyms
                        replace_to = [x.lower() for x in replace_to]
                    replace_to = random.choice(replace_to)

                    replace_from = None
                    for x in replace_from_cand:
                        if x in premise and x in hypothesis:
                            replace_from = x
                            break

                    if replace_from is None:
                        print(f"premise: {premise}")
                        print(f"hypothesis: {hypothesis}")
                        print(f"replace_from_cand: {replace_from_cand}")
                        raise ValueError("Could not find entity in the premise and hypothesis!")

                    aug["premise"] = premise.replace(replace_from, replace_to)
                    aug["hypothesis_string"] = hypothesis.replace(replace_from, replace_to)

                    augmented.append(aug)
            else:  # tail
                for _ in range(args.augment_num):
                    aug = row.to_dict()
                    aug["augmentation"] = "replace_tail"

                    replace_from_cand = [tail.name] + tail.synonyms
                    replace_from_cand = [x.lower() for x in replace_from_cand]
                    if tail.type == "species_with_part":
                        replace_from_cand = [x.split(" - ")[0] for x in replace_from_cand]
                    replace_from_cand.sort(key=len, reverse=True)
                    replace_to = replace_from_cand.copy()

                    if "chemical" in tail.type:
                        df_sampling_pool = df_chemical
                    else:
                        raise ValueError()

                    while set(replace_to) == set(replace_from_cand):
                        replace_to = df_sampling_pool.sample(n=1).iloc[0]
                        replace_to = [replace_to.name] + replace_to.synonyms
                        replace_to = [x.lower() for x in replace_to]
                    replace_to = random.choice(replace_to)

                    replace_from = None
                    for x in replace_from_cand:
                        if x in premise and x in hypothesis:
                            replace_from = x
                            break

                    if replace_from is None:
                        print(f"premise: {premise}")
                        print(f"hypothesis: {hypothesis}")
                        print(f"replace_from_cand: {replace_from_cand}")
                        raise ValueError("Could not find entity in the premise and hypothesis!")

                    aug["premise"] = premise.replace(replace_from, replace_to)
                    aug["hypothesis_string"] = hypothesis.replace(replace_from, replace_to)

                    augmented.append(aug)

        df_train = pd.DataFrame(augmented)

    df_train = df_train[["premise", "hypothesis_string", "hypothesis_id", "answer", "augmentation"]]
    df_train.to_csv(
        args.train_filepath.replace('.tsv', f'_{args.round}.tsv'),
        sep='\t',
        index=False
    )


def main():
    args = parse_argument()

    post_annotation_filepath = args.post_annotation_filepath.replace('*', str(args.round))
    df_annotated = read_annotated(post_annotation_filepath)

    if args.random_state:
        random.seed(args.random_state)

    print("Exporting post annotation data to KG...")
    fa_kg = export_to_kg(df_annotated, args)

    sys.exit()

    print("Generating training data...")
    generate_training(df_annotated, fa_kg, args)


if __name__ == '__main__':
    main()
