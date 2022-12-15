import argparse
import os
import requests
import sys
import warnings

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph,  CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RETIRED_ENTITIES_FILENAME = "retired_entities.txt"
RELATIONS_FILENAME = "relations.txt"
COMPATIBLE_FOOD_DBS = {"ncbi_taxid": "NCBI_taxonomy"}
COMPATIBLE_CHEMICAL_DBS = {"pubchem": "PubChem", "cas": "CAS"}


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

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
        "--external_db_filepath",
        type=str,
        required=True,
        help="Filepath of external DB.",
    )

    parser.add_argument(
        "--external_db_name",
        type=str,
        required=True,
        help="Name of external DB.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    #
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(args.input_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.input_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.input_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.input_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
    )

    #
    df_to_add = pd.read_csv(args.external_db_filepath, sep='\t', keep_default_na=False)
    columns = list(df_to_add.columns)
    print(f"Columns: {columns}")

    data = []
    for _, row in df_to_add.iterrows():
        row_dict = {k: v for k, v in row.to_dict().items() if v != ''}

        compatible_food_columns = []
        for x in row_dict.keys():
            if x.startswith("head_") and x.replace("head_", "") in COMPATIBLE_FOOD_DBS.keys():
                compatible_food_columns.append(x)

        if len(compatible_food_columns) == 0:
            warnings.warn("No compatible food columns found!")
            continue

        compatible_chemical_db_columns = []
        for x in row_dict.keys():
            if x.startswith("tail_") and x.replace("tail_", "") in COMPATIBLE_CHEMICAL_DBS.keys():
                compatible_chemical_db_columns.append(x)

        if len(compatible_chemical_db_columns) == 0:
            warnings.warn("No compatible chemical DB columns found!")
            continue

        head = CandidateEntity(
            type="organism",
            name=row_dict["head"],
            other_db_ids={v: [str(row_dict[f"head_{k}"])] for k, v in COMPATIBLE_FOOD_DBS.items()}
        )

        relation = CandidateRelation(
            name='contains',
            translation='contains',
        )

        other_db_ids = {
            v: [str(row_dict[f"tail_{k}"])]
            for k, v in COMPATIBLE_CHEMICAL_DBS.items()
            if f"tail_{k}" in row_dict
        }
        tail = CandidateEntity(
            type="chemical",
            name=row_dict["tail"],
            other_db_ids=other_db_ids
        )

        if "reference_pmid" in row_dict:
            pmid = str(int(row_dict["reference_pmid"]))
        else:
            pmid = ""

        if "reference_title" in row_dict:
            title = row_dict["reference_title"]
        else:
            title = ""

        data.append([head, relation, tail, pmid, title, row_dict["quality"]])

    df_candiate_triples = pd.DataFrame(
        data, columns=["head", "relation", "tail", "pmid", "title", "quality"])
    df_candiate_triples["source"] = args.external_db_name

    print(df_candiate_triples)
    sys.exit()

    fa_kg.add_triples(df_candiate_triples)

    fa_kg.save(
        kg_filepath=os.path.join(args.output_kg_dir, KG_FILENAME),
        evidence_filepath=os.path.join(args.output_kg_dir, EVIDENCE_FILENAME),
        entities_filepath=os.path.join(args.output_kg_dir, ENTITIES_FILENAME),
        retired_entities_filepath=os.path.join(args.output_kg_dir, RETIRED_ENTITIES_FILENAME),
        relations_filepath=os.path.join(args.output_kg_dir, RELATIONS_FILENAME),
    )


if __name__ == "__main__":
    main()
