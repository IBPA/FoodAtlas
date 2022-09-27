import argparse
import os
import sys

sys.path.append('..')

from common_utils.utils import read_tsv  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


PREDICTED_FILEPATH = "../../outputs/data_generation/predicted_{}.tsv"
THRESHOLD = 0.5
KG_OUTPUT_DIR = "../../outputs/kg"
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="What pre_annotation round are we generating.",
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

    # read predicted
    predicted_filepath = PREDICTED_FILEPATH.format(args.round)
    df_pred = read_tsv(predicted_filepath)
    df_pred["source"] = "prediction:round_1"
    print(f"Predicted shape: {df_pred.shape}")

    df_pred_pos = df_pred[df_pred["prob"] > args.threshold]
    print(f"Predicted positives shape: {df_pred_pos.shape}")

    # add predictions
    output_dir = os.path.join(KG_OUTPUT_DIR, str(args.round))
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(output_dir, KG_FILENAME),
        evidence_filepath=os.path.join(output_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(output_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(output_dir, RELATIONS_FILENAME),
    )
    fa_kg.add_ph_pairs(df_pred_pos)

    fa_kg.save()


if __name__ == '__main__':
    main()
