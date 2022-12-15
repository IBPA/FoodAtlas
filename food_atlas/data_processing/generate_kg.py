import argparse
import os
from pathlib import Path
import sys

from common_utils.utils import read_dataframe
from common_utils.knowledge_graph import KnowledgeGraph

THRESHOLD = 0.5
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KG from either annotations or predictions.")

    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Filepath of the file used to generate the KG."
    )

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--output_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Are you adding annotated positives of predictions (annotated|predicted).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Threshold cutoff (Default: {THRESHOLD})",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if args.mode == "annotated":
        df = read_dataframe(args.input_filepath)
        print(f"Input file shape: {df.shape}")

        df_pos = df[df["answer"] == "Entails"]
        df_pos["quality"] = "high"
        print(f"Positives shape: {df_pos.shape}")
    elif args.mode == "predicted":
        df = read_dataframe(args.input_filepath)
        df["source"] = "prediction:entailment"
        print(f"Predicted shape: {df.shape}")

        df_pos = df[df["prob_mean"] > args.threshold]
        df_pos["quality"] = "predicted"
        print(f"Predicted positives shape: {df_pos.shape}")
    else:
        raise ValueError()

    df_pos["title"] = ""

    # add predictions
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.input_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.input_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    fa_kg.add_ph_pairs(df_pos)

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.output_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == '__main__':
    main()
