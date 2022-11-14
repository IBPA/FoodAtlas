import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

THRESHOLD = 0.5
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
TRAIN_FILENAME = "train.txt"
VAL_FILENAME = "val.txt"
TEST_FILENAME = "test.txt"

COLUMNS_TO_KEEP = ["head", "relation", "tail"]


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the train/val/test data for KGC.")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="Directory containing the KG generated by the active learning loop.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save train/val/test set.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.input_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    df_relations = fa_kg.get_all_relations()
    contains_foodatlas_id = df_relations[
        df_relations["name"] == "contains"]["foodatlas_id"].tolist()[0]

    # extract train/va/test from evidence
    df_evidence = pd.read_csv(
        os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        sep='\t',
        keep_default_na=False,
    )

    df_train = df_evidence[df_evidence["source"].apply(
        lambda x: x not in ["annotation:val", "annotation:test"])]
    df_val = df_evidence[df_evidence["source"] == "annotation:val"]
    df_test = df_evidence[df_evidence["source"] == "annotation:test"]

    df_val = df_val[df_val["relation"] == contains_foodatlas_id]
    df_test = df_test[df_test["relation"] == contains_foodatlas_id]

    df_train = df_train[COLUMNS_TO_KEEP]
    df_val = df_val[COLUMNS_TO_KEEP]
    df_test = df_test[COLUMNS_TO_KEEP]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(os.path.join(args.output_dir, TRAIN_FILENAME), sep='\t', index=False)
    df_val.to_csv(os.path.join(args.output_dir, VAL_FILENAME), sep='\t', index=False)
    df_test.to_csv(os.path.join(args.output_dir, TEST_FILENAME), sep='\t', index=False)

    # copy other files
    shutil.copy(
        os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        os.path.join(args.output_dir, ENTITIES_FILENAME)
    )

    shutil.copy(
        os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
        os.path.join(args.output_dir, RELATIONS_FILENAME)
    )

    # generate hypotheses


if __name__ == '__main__':
    main()
