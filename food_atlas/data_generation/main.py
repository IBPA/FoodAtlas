import argparse
from datetime import datetime
from itertools import product
from pathlib import Path
import re
import requests
import time
from typing import Dict, List, Optional
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402

<<<<<<< HEAD
from common_utils.foodatlas_types import FoodAtlasEntity, FoodAtlasRelation  # noqa: E402
=======
>>>>>>> dev_local
from common_utils.foodatlas_types import CandidateEntity, CandidateRelation  # noqa: E402


FOOD_NAMES_FILEPATH = "../../data/FooDB/foodb_foods.txt"
QUERY_FSTRING = "{} contains"
QUERY_RESULTS_FILEPATH = "../../output/data_generation/query_results.txt"
FOOD_PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"
HYPOTHESES_FILEPATH = "../../output/data_generation/hypotheses_{}.txt"
ENTITIES_FILEPATH = "../../data/FoodAtlas/entities.txt"
RELATIONS_FILEPATH = "../../data/FoodAtlas/relations.txt"


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main code for running the data generation.")

    parser.add_argument(
        "--food_names_filepath",
        type=str,
        default=FOOD_NAMES_FILEPATH,
        help=f"File containing query food names. (Default: {FOOD_NAMES_FILEPATH})",
    )

    parser.add_argument(
        "--query_fstring",
        type=str,
        default=QUERY_FSTRING,
        help=f"fstring to format the query. (Default: {QUERY_FSTRING})",
    )

    parser.add_argument(
        "--query_results_filepath",
        type=str,
        default=QUERY_RESULTS_FILEPATH,
        help=f"Filepath to save the query results. (Default: {QUERY_RESULTS_FILEPATH})",
    )

    parser.add_argument(
        "--food_parts_filepath",
        type=str,
        default=FOOD_PARTS_FILEPATH,
        help=f"Filepath to food parts. (Default: {FOOD_PARTS_FILEPATH})",
    )

    parser.add_argument(
        "--overwrite_query_results",
        action="store_true",
        help="If set, overwrite the query results specified at --query_results_filepath.",
    )

    parser.add_argument(
        "--num_skip_query_lines",
        type=int,
        default=0,
        help="Number of queries to skip. (Default: 0)",
    )

    parser.add_argument(
        "--hypotheses_filepath",
        type=str,
        default=HYPOTHESES_FILEPATH,
        help=f"Filepath to save the query results. (Default: {HYPOTHESES_FILEPATH})",
    )

    parser.add_argument(
        "--entities_filepath",
        type=str,
        default=ENTITIES_FILEPATH,
        help=f"Filepath to save the entities. (Default: {ENTITIES_FILEPATH})",
    )

    parser.add_argument(
        "--relations_filepath",
        type=str,
        default=RELATIONS_FILEPATH,
        help=f"Filepath to save the entities. (Default: {RELATIONS_FILEPATH})",
    )

    args = parser.parse_args()
    return args


def get_food_parts_from_premise(
    premise: str,
    df_food_parts: pd.DataFrame,
) -> List[CandidateEntity]:
    premise = re.sub('[^A-Za-z0-9 ]+', '', premise.lower()).split()

    parts = []
    for idx, row in df_food_parts.iterrows():
        synonyms = row.food_part_synonyms.split(', ')
        parts_list = [row.food_part] + synonyms

        for part in parts_list:
            if part.lower() in premise:
                part_entity = CandidateEntity(
                    type="food_part",
                    name=row.food_part,
                    synonyms=synonyms,
                )
                parts.append(part_entity)

    return parts


def query_litsense(
    food_names: List[str],
    query_results_filepath: str,
    df_food_parts: pd.DataFrame,
    query_fstring: str,
    num_skip_query_lines: int,
    overwrite_query_results: bool,
    entities_filepath: str,
    relations_filepath: str,
    delay: float = 1.0,
) -> Optional[pd.DataFrame]:
    # make output dir
    query_results_dir = "/".join(query_results_filepath.split("/")[:-1])
    Path(query_results_dir).mkdir(parents=True, exist_ok=True)

    # check for existing file
    if Path(query_results_filepath).is_file() and not overwrite_query_results:
        # query results exist
        print("Query results already exists. Updating it.")
        df_existing = pd.read_csv(query_results_filepath, sep='\t')
    else:
        print("Writing a new query results.")
        df_existing = None

    if num_skip_query_lines > 0:
        print(f"Skipping {num_skip_query_lines} lines...")
        food_names = food_names[num_skip_query_lines:]

    data = []
    for idx, food_name in enumerate(food_names):
        print(f"Processing {idx+1}/{len(food_names)}...")

        try:
            search_term = query_fstring.format(food_name)
            print(f"Requesting data: {search_term}")

            # avoid throttling
            time.sleep(delay)

            query_url = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/" + \
                        f"?query={search_term}&rerank=true"

            response = requests.get(query_url)
            if response.status_code != 200:
                raise ValueError(f"Error requesting data from {query_url}: {response.status_code}")

            rows = []
            print("{} results recieved".format(len(response.json())))
            print("Parsing results...")
            for doc in response.json():
                if doc["annotations"] is None:
                    continue

                row = {}
                row["search_term"] = search_term
                row["pmid"] = doc["pmid"]
                row["pmcid"] = doc["pmcid"]
                row["section"] = doc["section"]
                row["premise"] = doc["text"]
                chemicals = []
                species = []

                for ent in doc["annotations"]:
                    ent_split_results = ent.split("|")
                    if len(ent_split_results) != 4:
                        print(f"Unable to parse annotation: {ent}")
                        continue

                    start, n_chars, category, ent_id = ent_split_results

                    if ent_id == "None" or ent_id == "-":
                        print(f"Skipping entities with no ID: {ent}")
                        continue

                    start = int(start)
                    n_chars = int(n_chars)
                    ent_text = doc["text"][start:(start + n_chars)]

                    if category == "chemical":
                        candidate_ent = CandidateEntity(
                            type="chemical",
                            name=ent_text,
                            other_db_ids={"MESH": ent_id}
                        )
                        chemicals.append(candidate_ent)
                    elif category == "species":
                        candidate_ent = CandidateEntity(
                            type="species",
                            name=ent_text,
                            other_db_ids={"NCBI_taxonomy": ent_id}
                        )
                        species.append(candidate_ent)

                if len(chemicals) == 0 or len(species) == 0:
                    continue

                row["chemicals"] = str(chemicals)
                row["species"] = str(species)
                row["food_parts"] = str(get_food_parts_from_premise(doc["text"], df_food_parts))
                rows.append(row)

            data.extend(rows)
        except Exception as e:
            print(e)

    if not data:
        print("Data empty. Nothing to write.")
        return None

    df_new = pd.DataFrame(data)

    if df_existing is not None:
        df_final = pd.concat([df_existing, df_new])
    else:
        df_final = df_new

    df_final.drop_duplicates(inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    df_final.to_csv(query_results_filepath, sep='\t', index=False)

    return df_final


def generate_hypotheses(
    df: pd.DataFrame,
    hypotheses_filepath: str,
):
<<<<<<< HEAD
    df = df.copy().head(20)

=======
>>>>>>> dev_local
    def _read_foodatlasent(x):
        return eval(x, globals())
    df["chemicals"] = df["chemicals"].apply(_read_foodatlasent)
    df["species"] = df["species"].apply(_read_foodatlasent)
    df["food_parts"] = df["food_parts"].apply(_read_foodatlasent)

<<<<<<< HEAD
    print(df)
    sys.exit()

    def _f(row):
        print(row)
        newrows = []
        for s, p, c in product(row["species"], row["food_parts"], row["chemicals"]):
            print(s)
            print(p)
            print(c)
            print()
            newrow = row.copy()
            newrow["head"] = f"{s.name} {p.name}"
            newrow["relation"] = "contains"
            newrow["tail"] = f"{c.name}"
            newrow["head_type"] = "food_part"
            newrow["tail_type"] = "chemical"
            newrows.append(newrow)
=======
    species_contains_chemical = CandidateRelation(
        name='contains',
        translation='contains',
        head_type='species',
        tail_type='chemical',
    )

    species_with_part_contains_chemical = CandidateRelation(
        name='contains',
        translation='contains',
        head_type='species_with_part',
        tail_type='chemical',
    )

    has_part = CandidateRelation(
        name='hasPart',
        translation='has part',
        head_type='species',
        tail_type='species_with_part',
    )

    def _f(row):
        newrows = []

        if not row["food_parts"]:
            for s, c in product(row["species"], row["chemicals"]):
                newrow = row.copy().drop(["chemicals", "species", "food_parts"])
                newrow["head"] = s
                newrow["relation"] = species_contains_chemical
                newrow["tail"] = c
                newrows.append(newrow)
        else:
            for s, p, c in product(row["species"], row["food_parts"], row["chemicals"]):
                # contains
                newrow = row.copy().drop(["chemicals", "species", "food_parts"])

                species_with_part = CandidateEntity(
                    type="species_with_part",
                    name=f"{s.name} {p.name}"
                )

                newrow["head"] = species_with_part
                newrow["relation"] = species_with_part_contains_chemical
                newrow["tail"] = c
                newrows.append(newrow)

                # hasPart
                newrow = row.copy().drop(["chemicals", "species", "food_parts"])

                newrow["head"] = s
                newrow["relation"] = has_part
                newrow["tail"] = species_with_part
                newrows.append(newrow)
>>>>>>> dev_local

        return newrows

    results = []
    for result in df.parallel_apply(_f, axis=1):
        results.append(pd.DataFrame(result))

    df_hypotheses = pd.concat(results).reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_hypotheses.to_csv(hypotheses_filepath.format(timestamp), index=False, sep="\t")

    return df_hypotheses


def get_food_names_and_parts(
        food_names_filepath: str,
        food_parts_filepath: str,
        entities_filepath: str,
        relations_filepath: str,
):
    # food names
    if "data/FooDB" in food_names_filepath:
        data_source = "foodb"
    else:
        raise NotImplementedError()

    if data_source == "foodb":
        df_food_names = pd.read_csv(food_names_filepath, sep='\t', keep_default_na=False)
        food_names = [x for x in df_food_names["food_name"].tolist() if x != ""]
        synonyms = [x for x in df_food_names["food_name_synonyms"].tolist() if x != ""]
        food_names += synonyms
        food_names = list(set(food_names))

    print(f"Found {len(food_names)} food names to use for query.")

    # food parts
    df_food_parts = pd.read_csv(food_parts_filepath, sep='\t', keep_default_na=False)

    return food_names, df_food_parts


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    # food names and parts
    food_names, df_food_parts = get_food_names_and_parts(
        food_names_filepath=args.food_names_filepath,
        food_parts_filepath=args.food_parts_filepath,
        entities_filepath=args.entities_filepath,
        relations_filepath=args.relations_filepath,
    )

    # df = query_litsense(
    #     food_names=food_names,
    #     query_results_filepath=args.query_results_filepath,
    #     df_food_parts=df_food_parts,
    #     query_fstring=args.query_fstring,
    #     num_skip_query_lines=args.num_skip_query_lines,
    #     overwrite_query_results=args.overwrite_query_results,
    #     entities_filepath=args.entities_filepath,
    #     relations_filepath=args.relations_filepath,
    # )

    df = pd.read_csv(args.query_results_filepath, sep='\t')

    generate_hypotheses(
        df=df,
        hypotheses_filepath=args.hypotheses_filepath,
    )


if __name__ == "__main__":
    main()
