import argparse
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
import re
import requests
import time
from typing import List, Optional
import sys

import pandas as pd
from pandarallel import pandarallel

from common_utils.knowledge_graph import CandidateEntity, CandidateRelation
from common_utils.knowledge_graph import KnowledgeGraph


FOOD_NAMES_FILEPATH = "../../data/FooDB/foodb_foods.txt"
QUERY_FSTRING = "{} contains"
QUERY_RESULTS_FILEPATH = "../../outputs/data_generation/query_results.txt"
FOOD_PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"
PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_{}.txt"
ENTITIES_FILEPATH = "../../data/FoodAtlas/entities.txt"
RELATIONS_FILEPATH = "../../data/FoodAtlas/relations.txt"


def parse_argument() -> argparse.Namespace:
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
        "--ph_pairs_filepath",
        type=str,
        default=PH_PAIRS_FILEPATH,
        help=f"Filepath to save the PH pairs. (Default: {PH_PAIRS_FILEPATH})",
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


def get_food_parts(
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

                if part_entity not in parts:
                    parts.append(part_entity)

    return parts


def query_litsense(
    df_food_names: pd.DataFrame,
    query_results_filepath: str,
    df_food_parts: pd.DataFrame,
    query_fstring: str,
    overwrite_query_results: bool,
    entities_filepath: str,
    relations_filepath: str,
    delay: float = 1.0,
) -> Optional[pd.DataFrame]:
    # make outputs dir
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

    ncbi_taxonomy_ids = list(set(df_food_names["ncbi_taxonomy_id"].tolist()))

    data = []
    for idx, row in df_food_names.iterrows():
        print(f"Processing {idx+1}/{df_food_names.shape[0]}...")

        food_names = [row["name"], row["name_scientific"]]
        food_names = [x for x in food_names if x != ""]

        for food_name in food_names:
            search_term = query_fstring.format(food_name)
            print(f"Requesting data: {search_term}")

            # avoid throttling
            time.sleep(delay)

            query_url = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/" + \
                        f"?query={search_term}&rerank=true"

            response = requests.get(query_url)
            if response.status_code != 200:
                raise ValueError(
                    f"Error requesting data from {query_url}: {response.status_code}")

            data_to_extend = []
            print("{} results recieved".format(len(response.json())))
            print("Parsing results...")
            for doc in response.json():
                if doc["annotations"] is None:
                    continue

                r = {}
                r["search_term"] = search_term
                r["pmid"] = doc["pmid"]
                r["pmcid"] = doc["pmcid"]
                r["section"] = doc["section"]
                r["premise"] = doc["text"]
                chemicals = []
                organisms = []

                for ent in doc["annotations"]:
                    ent_split_results = ent.split("|")
                    if len(ent_split_results) != 4:
                        print(f"Unable to parse annotation: {ent}")
                        continue

                    start, n_chars, category, ent_id = ent_split_results
                    start = int(start)
                    n_chars = int(n_chars)

                    if start < 0:
                        print("Start position cannot be less than 0.")
                        continue

                    ent_text = doc["text"][start:(start + n_chars)]

                    if ent_text == "":
                        print(f"Skipping empty entity (start: {start}, n_chars: {n_chars})")
                        continue

                    if ent_id == "None" or ent_id == "-":
                        print(f"Skipping entities with no ID: {ent_text}")
                        continue

                    if category == "species" and not ent_id.isdigit():
                        print(f"Skipping species with non-numerical ID: {ent_id}.")
                        continue

                    if category == "species" and int(ent_id) not in ncbi_taxonomy_ids:
                        print(f"Skipping entity with ID not in FooDB: {ent_text} ({ent_id})")
                        continue

                    if category == "chemical":
                        candidate_ent = CandidateEntity(
                            type="chemical",
                            name=ent_text,
                            other_db_ids={"MESH": ent_id.replace("MESH:", "")}
                        )
                        chemicals.append(candidate_ent)
                    elif category == "species":
                        match = df_food_names[
                            df_food_names["ncbi_taxonomy_id"] == int(ent_id)].iloc[0]
                        synonyms = [ent_text, match["name_scientific"]]
                        synonyms = [x for x in synonyms
                                    if x.lower() != match["name"].lower() and x != ""]
                        synonyms = list(set(synonyms))

                        candidate_ent = CandidateEntity(
                            type="organism",
                            name=match["name"],
                            synonyms=synonyms,
                            other_db_ids={"NCBI_taxonomy": ent_id}
                        )
                        organisms.append(candidate_ent)

                if len(chemicals) == 0 or len(organisms) == 0:
                    continue

                # clean up the entities
                chemicals = KnowledgeGraph.merge_candidate_entities(
                    chemicals, using="MESH")
                organisms = KnowledgeGraph.merge_candidate_entities(
                    organisms, using="NCBI_taxonomy")

                r["chemicals"] = str(chemicals)
                r["organisms"] = str(organisms)
                r["food_parts"] = str(get_food_parts(doc["text"], df_food_parts))
                data_to_extend.append(r)

            data.extend(data_to_extend)

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


def generate_ph_pairs(
    df: pd.DataFrame,
    ph_pairs_filepath: str,
):
    df["chemicals"] = df["chemicals"].apply(lambda x: eval(x, globals()))
    df["organisms"] = df["organisms"].apply(lambda x: eval(x, globals()))
    df["food_parts"] = df["food_parts"].apply(lambda x: eval(x, globals()))

    contains = CandidateRelation(
        name='contains',
        translation='contains',
    )

    def _f(row):
        newrows = []
        failed = []

        cleaned_premise = " " + re.sub('[^A-Za-z0-9 ]+', ' ', row["premise"].lower()) + ""

        for s, c in product(row["organisms"], row["chemicals"]):
            newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])
            newrow["head"] = s
            newrow["relation"] = contains
            newrow["tail"] = c

            organisms = None
            if s.name.lower() not in row["premise"].lower() or \
               f" {s.name.lower()} " not in cleaned_premise:
                for x in s.synonyms:
                    if x.lower() in row["premise"].lower():
                        organisms = x
            else:
                organisms = s.name

            chemicals = None
            if c.name.lower() not in row["premise"].lower():
                for x in c.synonyms:
                    if x.lower() in row["premise"].lower():
                        chemicals = x
            else:
                chemicals = c.name

            if organisms is None or chemicals is None:
                failed.append(row)
                continue

            newrow["hypothesis_string"] = f"{organisms} contains {chemicals}"

            ncbi_taxonomy = s.other_db_ids["NCBI_taxonomy"]
            mesh = c.other_db_ids["MESH"]
            newrow["hypothesis_id"] = f"NCBI_taxonomy:{ncbi_taxonomy}_contains_MESH:{mesh}"
            newrows.append(newrow)

        if row["food_parts"]:
            for s, p, c in product(row["organisms"], row["food_parts"], row["chemicals"]):
                # contains
                newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])

                organism_with_part = CandidateEntity(
                    type="organism_with_part",
                    name=f"{s.name} - {p.name}",
                    synonyms=[f"{x} - {p.name}" for x in s.synonyms],
                    other_db_ids=s.other_db_ids,
                )

                newrow["head"] = organism_with_part
                newrow["relation"] = contains
                newrow["tail"] = c

                organisms = None
                if s.name.lower() not in row["premise"].lower() or \
                   f" {s.name.lower()} " not in cleaned_premise:
                    for x in s.synonyms:
                        if x.lower() in row["premise"].lower():
                            organisms = x
                else:
                    organisms = s.name

                parts = None
                if p.name.lower() not in row["premise"].lower():
                    for x in p.synonyms:
                        if x.lower() in row["premise"].lower():
                            parts = x
                else:
                    parts = p.name

                chemicals = None
                if c.name.lower() not in row["premise"].lower():
                    for x in c.synonyms:
                        if x.lower() in row["premise"].lower():
                            chemicals = x
                else:
                    chemicals = c.name

                if organisms is None or parts is None or chemicals is None:
                    failed.append(row)
                    continue

                newrow["hypothesis_string"] = f"{organisms} - {parts} contains {chemicals}"

                ncbi_taxonomy = s.other_db_ids["NCBI_taxonomy"]
                mesh = c.other_db_ids["MESH"]
                newrow["hypothesis_id"] = f"NCBI_taxonomy:{ncbi_taxonomy}-" + \
                                          f"{p.name}_contains_MESH:{mesh}"
                newrows.append(newrow)

        return newrows, failed

    results = []
    skipped = []
    for result, failed in df.parallel_apply(_f, axis=1):
        results.append(pd.DataFrame(result))
        skipped.extend(failed)

    print(f"Skipped {len(skipped)} rows.")
    for x in skipped:
        print(f"Premise: {x['premise']}")
        print(f"Chemicals: {x['chemicals']}")
        print(f"Organisms: {x['organisms']}")
        print(f"Food parts: {x['food_parts']}")

    df_ph_pairs = pd.concat(results).reset_index(drop=True)
    df_ph_pairs.fillna("", inplace=True)
    df_ph_pairs = df_ph_pairs.astype(str)
    df_ph_pairs.drop_duplicates(inplace=True)

    hypothesis_id = df_ph_pairs["hypothesis_id"].tolist()
    duplicates = [item for item, count in Counter(hypothesis_id).items() if count > 1]
    print(f"Found {len(duplicates)} duplicate hypothesis IDs out of {len(hypothesis_id)}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_ph_pairs.to_csv(ph_pairs_filepath.format(timestamp), index=False, sep="\t")

    return df_ph_pairs


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

    # food parts
    df_food_parts = pd.read_csv(food_parts_filepath, sep='\t', keep_default_na=False)

    return df_food_names, df_food_parts


def sample_val_test_set(
    df_ph_pairs: pd.DataFrame,
    val_filepath: str,
    test_filepath: str,
    val_num_premise: float,
    test_num_premise: float,
):
    print(df_ph_pairs)


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    # food names and parts
    df_food_names, df_food_parts = get_food_names_and_parts(
        food_names_filepath=args.food_names_filepath,
        food_parts_filepath=args.food_parts_filepath,
        entities_filepath=args.entities_filepath,
        relations_filepath=args.relations_filepath,
    )

    # df = query_litsense(
    #     df_food_names=df_food_names,
    #     query_results_filepath=args.query_results_filepath,
    #     df_food_parts=df_food_parts,
    #     query_fstring=args.query_fstring,
    #     overwrite_query_results=args.overwrite_query_results,
    #     entities_filepath=args.entities_filepath,
    #     relations_filepath=args.relations_filepath,
    # )

    df = pd.read_csv(args.query_results_filepath, sep='\t')
    generate_ph_pairs(
        df=df,
        ph_pairs_filepath=args.ph_pairs_filepath,
    )


if __name__ == "__main__":
    main()
