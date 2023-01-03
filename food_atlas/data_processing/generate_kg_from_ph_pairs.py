import argparse
from copy import deepcopy
import sys

import pandas as pd
from tqdm import tqdm

from common_utils.chemical_db_ids import get_mesh_name_using_mesh_id, read_pubchem_cid_mesh
from common_utils.chemical_db_ids import read_mesh_data, get_pubchem_id_other_db_ids_using_mesh
from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.utils import read_dataframe


THRESHOLD = 0.5


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KG from either annotations or predictions PH paiirs.")

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

    parser.add_argument(
        "--new_pkl",
        action="store_true",
        help="Set if using new pickled data.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # read data
    if args.mode == "annotated":
        df = read_dataframe(args.input_filepath)
        print(f"Input file shape: {df.shape}")

        df_pos = df[df["answer"] == "Entails"]
        df_pos["source"] = "FoodAtlas:annotation"
        df_pos["quality"] = "high"
        print(f"Positives shape: {df_pos.shape}")
    elif args.mode == "predicted":
        df = read_dataframe(args.input_filepath)
        df["source"] = "FoodAtlas:prediction:entailment"
        print(f"Predicted shape: {df.shape}")

        df_pos = df[df["prob_mean"] > args.threshold]
        df_pos["quality"] = "low"
        print(f"Predicted positives shape: {df_pos.shape}")
    else:
        raise ValueError()

    df_pos["title"] = ""

    # find MeSH IDs in the data
    mesh_ids = []
    entities = df_pos["head"].tolist() + df_pos["tail"].tolist()
    for e in entities:
        if "MESH" in e.other_db_ids:
            mesh_ids.extend(e.other_db_ids["MESH"])
    mesh_ids = list(set(mesh_ids))
    print(f"Found {len(mesh_ids)} MeSH IDs")

    # find the Dict[pubchem ID, other_db_ids] using the MeSH IDs
    pubchem_id_other_db_ids_dict = get_pubchem_id_other_db_ids_using_mesh(
        mesh_ids, args.new_pkl)

    # now update ph_pairs
    mesh_data_dict = read_mesh_data(use_pkl=True)
    df_cid_mesh = read_pubchem_cid_mesh()

    rows = []
    print("Updating chemical entities in the PH pairs...")
    for _, row in tqdm(df_pos.iterrows(), total=df_pos.shape[0]):
        head = row["head"]
        tail = row["tail"]

        if "MESH" in head.other_db_ids:
            raise NotImplementedError()

        assert "MESH" in tail.other_db_ids
        mesh_id = tail.other_db_ids["MESH"]
        assert len(mesh_id) == 1
        mesh_id = mesh_id[0]
        mesh_name = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
        if mesh_name is None:
            continue

        df_cid_mesh_match = df_cid_mesh[df_cid_mesh["mesh_name"].apply(lambda x: x == mesh_name)]
        pubchem_ids = list(set(df_cid_mesh_match["pubchem_id"].tolist()))
        if len(pubchem_ids) == 0:
            continue

        for pubchem_id in pubchem_ids:
            other_db_ids = pubchem_id_other_db_ids_dict[pubchem_id]
            if mesh_id not in other_db_ids["MESH"]:
                raise RuntimeError(f"Did not find {mesh_id} in {other_db_ids}")
            row["tail"] = tail._replace(other_db_ids=other_db_ids)
            rows.append(deepcopy(row))

    df_pos_updated = pd.DataFrame(rows)

    # add predictions
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    fa_kg.add_ph_pairs(df_pos_updated)
    fa_kg.save(kg_dir=args.output_kg_dir)


if __name__ == '__main__':
    main()
