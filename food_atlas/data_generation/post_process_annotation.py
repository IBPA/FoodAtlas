import argparse
import os
from pathlib import Path
import random
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.utils import read_annotated  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


RANDOM_STATE = 530
VAL_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_post_annotation.tsv"
TEST_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_post_annotation.tsv"
VAL_FILEPATH = "../../outputs/data_generation/val.tsv"
TEST_FILEPATH = "../../outputs/data_generation/test.tsv"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
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
        required=True,
        help="Annotation filepath.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Set random state.",
    )

    parser.add_argument(
        "--kg_output_dir",
        type=str,
        required=True,
        help="Directory to load/save the knowledge graph.",
    )

    parser.add_argument(
        "--prev_kg_output_dir",
        type=str,
        help="Directory to load the previous knowledge graph (required for round >= 2).",
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
        required=True,
        help="Train filepath.",
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
        "--do_augmentation",
        action="store_true",
        help="Set if performing augmentation.",
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
        # val and test need to be cleaned up
        df_val = read_annotated(args.val_post_annotation_filepath)
        df_test = read_annotated(args.test_post_annotation_filepath)
        df_val.to_csv(args.val_filepath, sep='\t', index=False)
        df_test.to_csv(args.test_filepath, sep='\t', index=False)

        df_val["source"] = "annotation:val"
        df_test["source"] = "annotation:test"

        df_val = df_val[df_annotated.columns]
        df_test = df_test[df_annotated.columns]

        df_for_kg = pd.concat([df_val, df_test, df_annotated]).reset_index(drop=True)

        Path(args.kg_output_dir).mkdir(parents=True, exist_ok=True)
        kg_filepath = os.path.join(args.kg_output_dir, KG_FILENAME)
        evidence_filepath = os.path.join(args.kg_output_dir, EVIDENCE_FILENAME)
        entities_filepath = os.path.join(args.kg_output_dir, ENTITIES_FILENAME)
        relations_filepath = os.path.join(args.kg_output_dir, RELATIONS_FILENAME)
    else:
        df_for_kg = df_annotated.copy()

        Path(args.kg_output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.prev_kg_output_dir).is_dir()
        kg_filepath = os.path.join(args.prev_kg_output_dir, KG_FILENAME)
        evidence_filepath = os.path.join(args.prev_kg_output_dir, EVIDENCE_FILENAME)
        entities_filepath = os.path.join(args.prev_kg_output_dir, ENTITIES_FILENAME)
        relations_filepath = os.path.join(args.prev_kg_output_dir, RELATIONS_FILENAME)

    df_pos = df_for_kg[df_for_kg["answer"] == "Entails"]
    df_pos.reset_index(inplace=True, drop=True)

    fa_kg = KnowledgeGraph(
        kg_filepath=kg_filepath,
        evidence_filepath=evidence_filepath,
        entities_filepath=entities_filepath,
        relations_filepath=relations_filepath,
    )
    fa_kg.add_ph_pairs(df_pos)

    if args.round == 1:
        fa_kg.save()
    else:
        fa_kg.save(
            kg_filepath=os.path.join(args.kg_output_dir, KG_FILENAME),
            evidence_filepath=os.path.join(args.kg_output_dir, EVIDENCE_FILENAME),
            entities_filepath=os.path.join(args.kg_output_dir, ENTITIES_FILENAME),
            relations_filepath=os.path.join(args.kg_output_dir, RELATIONS_FILENAME),
        )

    return fa_kg


def generate_training(df_annotated, fa_kg, args):
    df_train = df_annotated.copy()[
        ["head", "relation", "tail", "premise", "hypothesis_string", "hypothesis_id", "answer"]
    ]

    if not args.do_augmentation:
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
    Path(args.train_filepath).parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(args.train_filepath, sep='\t', index=False)


def main():
    args = parse_argument()

    if args.round >= 2 and args.prev_kg_output_dir is None:
        raise ValueError(f"--prev_kg_output_dir argument must be specified for round {args.round}")

    df_annotated = read_annotated(args.post_annotation_filepath)

    if args.random_state:
        random.seed(args.random_state)

    # print("Exporting post annotation data to KG...")
    # fa_kg = export_to_kg(df_annotated, args)

    print("Generating training data...")
    generate_training(df_annotated, None, args)


if __name__ == '__main__':
    main()
