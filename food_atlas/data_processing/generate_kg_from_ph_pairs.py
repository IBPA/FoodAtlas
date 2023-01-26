import argparse
from copy import deepcopy
import warnings
import sys

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from common_utils.chemical_db_ids import get_mesh_name_using_mesh_id, read_pubchem_cid_mesh
from common_utils.chemical_db_ids import read_mesh_data, get_pubchem_id_other_db_ids_using_mesh
from common_utils.knowledge_graph import KnowledgeGraph, CandidateRelation
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

    parser.add_argument(
        "--nb_workers",
        type=int,
        help="Number of workers for pandarallel.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if args.nb_workers is None:
        pandarallel.initialize(progress_bar=True)
    else:
        pandarallel.initialize(progress_bar=True, nb_workers=args.nb_workers)

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
        df_pos["quality"] = "medium"
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

    has_part = CandidateRelation(
        name='hasPart',
        translation='has part',
    )

    def _f(row):
        head = row["head"]
        tail = row["tail"]
        newrows = []

        if "MESH" in head.other_db_ids:
            raise NotImplementedError()

        assert "MESH" in tail.other_db_ids
        mesh_id = tail.other_db_ids["MESH"]
        assert len(mesh_id) == 1

        if 'MESH' not in tail.other_db_ids or len(mesh_id) != 1:
            print("MESH not in other_db_ids or len(mesn_id) != 1")
            raise RuntimeError()
        mesh_id = mesh_id[0]
        mesh_name = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
        if mesh_name is None:
            return newrows

        df_cid_mesh_match = df_cid_mesh[df_cid_mesh["mesh_name"].apply(lambda x: x == mesh_name)]
        pubchem_ids = list(set(df_cid_mesh_match["pubchem_id"].tolist()))
        if len(pubchem_ids) == 0:
            return newrows

        for pubchem_id in pubchem_ids:
            other_db_ids = pubchem_id_other_db_ids_dict[pubchem_id]
            if mesh_id not in other_db_ids["MESH"]:
                warnings.warn(f"Did not find {mesh_id} in {other_db_ids}")
                continue
            row["tail"] = tail._replace(other_db_ids=other_db_ids)
            newrows.append(deepcopy(row))

        # hasPart
        if head.type == "organism_with_part":
            head_without_part = deepcopy(head)
            new_other_db_ids = deepcopy(head_without_part.other_db_ids)
            new_other_db_ids["foodatlas_part_id"] = 'p0'
            head_without_part = head_without_part._replace(
                type=head_without_part.type.replace("organism_with_part", "organism"),
                name=head_without_part.name.split(" - ")[0],
                synonyms=[x.split(" - ")[0] for x in head_without_part.synonyms],
                other_db_ids=new_other_db_ids,
            )

            row["head"] = head_without_part
            row["relation"] = has_part
            row["tail"] = head
            newrows.append(row)

        return newrows

    #
    print("Updating chemical entities in the PH pairs...")
    results = []
    for result in df_pos.parallel_apply(_f, axis=1):
        results.extend(result)
    df_pos_updated = pd.DataFrame(results)

    # add predictions
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir, nb_workers=args.nb_workers)
    fa_kg.add_triples(df_pos_updated)
    fa_kg.save(kg_dir=args.output_kg_dir)


if __name__ == '__main__':
    main()
