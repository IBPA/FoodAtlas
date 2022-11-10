import argparse
import os
from pathlib import Path
import sys

sys.path.append('..')

from common_utils.utils import read_tsv  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


THRESHOLD = 0.5
KG_OUTPUT_DIR = "../../outputs/kg"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"

MODES = [
    "annotated",
    "predicted",
]


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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the KG files."
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

    df = read_tsv(args.input_filepath)
    print(f"Input file shape: {df.shape}")

    if args.mode == "annotated":
        df_pos = df[df["answer"] == "Entails"]
        print(f"Positives shape: {df_pos.shape}")
    elif args.mode == "predicted":
        raise NotImplementedError()
    else:
        raise ValueError()

    # add predictions
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.output_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_dir, RELATIONS_FILENAME),
    )
    fa_kg.add_ph_pairs(df_pos)

    fa_kg.save()


if __name__ == '__main__':
    main()
