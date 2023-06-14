import argparse
from ast import literal_eval
from collections import Counter
from copy import deepcopy
from datetime import datetime
import glob
import os
from itertools import product
import json
from pathlib import Path
import re
import requests
import time
from typing import List, Optional
import sys
import warnings

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from common_utils.knowledge_graph import CandidateEntity, CandidateRelation
from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.utils import save_pkl, load_pkl


QUERY_RESULTS_FILEPATH = "../../outputs/data_processing/query_results_{}.txt"
FOOD_PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"
PH_PAIRS_FILEPATH = "../../outputs/data_processing/ph_pairs_{}.txt"
FAILED_QUERY_FILENAME = "failed_query.pkl"
QUERY_DATA_PKL_FILENAME = "query_data.pkl"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main code for running the data generation.")

    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        help="Either query_string or query_results.",
    )

    parser.add_argument(
        "--query_filepath",
        type=str,
        required=True,
        help="Filepath for queries (text file with one query per line).",
    )

    parser.add_argument(
        "--food_parts_filepath",
        type=str,
        default=FOOD_PARTS_FILEPATH,
        help=f"Filepath to food parts. (Default: {FOOD_PARTS_FILEPATH})",
    )

    parser.add_argument(
        "--allowed_ncbi_taxids_filepath",
        type=str,
        required=True,
        help="Filepath to that contains alloed NCBI tax IDs.",
    )

    parser.add_argument(
        "--query_results_filepath",
        type=str,
        default=QUERY_RESULTS_FILEPATH,
        help=f"Filepath to save the query results. (Default: {QUERY_RESULTS_FILEPATH})",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Set temporary directory to cache in case of fail.",
    )

    parser.add_argument(
        "--match_all",
        action="store_true",
        help="Set if match all queries when quering LitSense.",
    )

    parser.add_argument(
        "--ph_pairs_filepath",
        type=str,
        default=PH_PAIRS_FILEPATH,
        help=f"Filepath to save the PH pairs. (Default: {PH_PAIRS_FILEPATH})",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        help=f"Number of workers for pandarallel.",
    )

    args = parser.parse_args()
    return args


def get_food_parts(
    premise: str,
    df_food_parts: pd.DataFrame,
) -> List[CandidateEntity]:
    premise = re.sub('[^A-Za-z0-9 ]+', '', premise.lower()).split()

    parts = []
    for _, row in df_food_parts.iterrows():
        synonyms = row.food_part_synonyms.split(', ')
        parts_list = [row.food_part] + synonyms

        for part in parts_list:
            if part.lower() in premise:
                part_entity = CandidateEntity(
                    type="food_part",
                    name=row.food_part,
                    synonyms=synonyms,
                    other_db_ids={"foodatlas_part_id": row["foodatlas_part_id"]},
                )

                if part_entity not in parts:
                    parts.append(part_entity)

    return parts


def query_litsense(
    input_type: str,
    query_filepath: str,
    food_parts_filepath: str,
    allowed_ncbi_taxids_filepath: str,
    query_results_filepath: str,
    cache_dir: str,
    match_all: bool,
    delay: float = 0.9,
) -> Optional[pd.DataFrame]:
    if input_type == "query_string":
        # queries
        query_items = pd.read_csv(query_filepath, sep='\t', keep_default_na=False, names=["data"])
        query_items = query_items["data"].tolist()
    elif input_type == "query_results":
        query_items = sorted(glob.glob(query_filepath))
    else:
        raise ValueError()

    # food parts
    df_food_parts = pd.read_csv(food_parts_filepath, sep='\t', keep_default_na=False)

    # allowed ncbi taxids
    df_allowed_taxids = pd.read_csv(allowed_ncbi_taxids_filepath, sep='\t', keep_default_na=False)
    ncbi_taxonomy_ids = list(set(df_allowed_taxids["ncbi_taxonomy_id"].tolist()))

    # make outputs dir
    query_results_dir = "/".join(query_results_filepath.split("/")[:-1])
    Path(query_results_dir).mkdir(parents=True, exist_ok=True)

    query_data_pkl_filepath = os.path.join(cache_dir, QUERY_DATA_PKL_FILENAME)
    Path(query_data_pkl_filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"In case of runtime fail, query data will be pickled to {query_data_pkl_filepath}")

    data = []
    # data = load_pkl(query_data_pkl_filepath)
    # query_items = query_items[400:]

    for idx, search_term in enumerate(tqdm(query_items)):
        if input_type == "query_string":
            query_url = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/" + \
                f"?query={search_term}&rerank=true"

            if match_all:
                query_url += "&match_all=true"
            else:
                query_url += "&match_all=false"

            print(f"\nRequesting: {query_url}...")

            try_count = 0
            while True:
                time.sleep(delay)  # avoid throttling
                response = requests.get(query_url)
                if response.status_code == 404:
                    warnings.warn(f"Failed to request data from {query_url}: {response.status_code}")
                    break
                elif response.status_code not in [404, 200]:
                    warnings.warn(f"Error requesting data from {query_url}: {response.status_code}")
                    print('Trying again...')
                    try_count += 1
                    if try_count == 10:
                        break
                else:
                    break

            if response.status_code == 404:
                continue

            if response.status_code not in [404, 200, 502]:
                save_pkl(data, query_data_pkl_filepath)
                sys.exit(1)

            if idx % 100 == 0:
                save_pkl(data, query_data_pkl_filepath)
                print(f"Saving temporarl data pickle (idx: {idx}) file to {query_data_pkl_filepath}")

            response = response.json()
        elif input_type == "query_results":
            with open(search_term) as _f:
                response = _f.readlines()[1:]
                assert len(response) == 1
                response = json.loads(response[0])

        data_to_extend = []
        print("Parsing {} results...".format(len(response)))
        for doc in response:
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
                    # print("Start position cannot be less than 0.")
                    continue

                ent_text = doc["text"][start:(start + n_chars)]

                if ent_text == "":
                    # print(f"Skipping empty entity (start: {start}, n_chars: {n_chars})")
                    continue

                if ent_id == "None" or ent_id == "-":
                    # print(f"Skipping entities with no ID: {ent_text}")
                    continue

                if category == "species" and not ent_id.isdigit():
                    # print(f"Skipping species with non-numerical ID: {ent_id}.")
                    continue

                if category == "species" and int(ent_id) not in ncbi_taxonomy_ids:
                    # print(f"Skipping entity with ID not in FooDB: {ent_text} ({ent_id})")
                    continue

                if category == "chemical":
                    candidate_ent = CandidateEntity(
                        type="chemical",
                        name=ent_text,
                        other_db_ids={"MESH": [ent_id.replace("MESH:", "")]}
                    )
                    if candidate_ent not in chemicals:
                        chemicals.append(candidate_ent)
                elif category == "species":
                    candidate_ent = CandidateEntity(
                        type="organism",
                        name=ent_text,
                        other_db_ids={"NCBI_taxonomy": [ent_id]}
                    )
                    if candidate_ent not in organisms:
                        organisms.append(candidate_ent)

            if len(chemicals) == 0 or len(organisms) == 0:
                continue

            r["chemicals"] = str(chemicals)
            r["organisms"] = str(organisms)
            r["food_parts"] = str(get_food_parts(doc["text"], df_food_parts))
            data_to_extend.append(r)

        data.extend(data_to_extend)

    if not data:
        print("Data empty. Nothing to write.")
        return None

    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(query_results_filepath, sep='\t', index=False)

    return df


def generate_ph_pairs(
    df: pd.DataFrame,
    ph_pairs_filepath: str,
):
    subset = list(df.columns.values)
    subset.remove("search_term")
    df.drop_duplicates(subset, inplace=True, ignore_index=True)

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

        for s, c in product(row["organisms"], row["chemicals"]):
            newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])

            #
            new_s = deepcopy(s)
            new_other_db_ids = new_s.other_db_ids
            new_other_db_ids["foodatlas_part_id"] = "p0"
            new_s = new_s._replace(other_db_ids=new_other_db_ids)

            newrow["head"] = new_s
            newrow["relation"] = contains
            newrow["tail"] = c

            if new_s.name not in row["premise"] or c.name not in row["premise"]:
                failed.append(row)
                continue
            newrow["hypothesis_string"] = f"{new_s.name} contains {c.name}"

            assert len(new_s.other_db_ids["NCBI_taxonomy"]) == 1
            assert len(c.other_db_ids["MESH"]) == 1
            newrows.append(newrow)

        if row["food_parts"]:
            for s, p, c in product(row["organisms"], row["food_parts"], row["chemicals"]):
                # contains
                newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])

                new_s = deepcopy(s)
                new_other_db_ids = new_s.other_db_ids
                new_other_db_ids["foodatlas_part_id"] = p.other_db_ids["foodatlas_part_id"]
                new_s = new_s._replace(other_db_ids=new_other_db_ids)
                organism_with_part = CandidateEntity(
                    type="organism_with_part",
                    name=f"{new_s.name} - {p.name}",
                    other_db_ids=new_s.other_db_ids,
                )

                newrow["head"] = organism_with_part
                newrow["relation"] = contains
                newrow["tail"] = c

                part_in_premise = False
                for x in [p.name] + p.synonyms:
                    if x in row["premise"].lower():
                        part_in_premise = True

                if new_s.name not in row["premise"] or \
                        c.name not in row["premise"] or \
                        part_in_premise is False:
                    failed.append(row)
                    continue

                newrow["hypothesis_string"] = f"{new_s.name} - {p.name} contains {c.name}"

                assert len(s.other_db_ids["NCBI_taxonomy"]) == 1
                assert len(c.other_db_ids["MESH"]) == 1
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
    df_ph_pairs = df_ph_pairs.astype(str)
    df_ph_pairs.drop_duplicates(inplace=True)
    df_ph_pairs.to_csv(ph_pairs_filepath, index=False, sep="\t")

    return df_ph_pairs


def main():
    args = parse_argument()

    if args.nb_workers is None:
        pandarallel.initialize(progress_bar=True)
    else:
        pandarallel.initialize(progress_bar=True, nb_workers=args.nb_workers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.query_results_filepath = args.query_results_filepath.format(timestamp)
    args.ph_pairs_filepath = args.ph_pairs_filepath.format(timestamp)

    df = query_litsense(
        input_type=args.input_type,
        query_filepath=args.query_filepath,
        food_parts_filepath=args.food_parts_filepath,
        allowed_ncbi_taxids_filepath=args.allowed_ncbi_taxids_filepath,
        query_results_filepath=args.query_results_filepath,
        cache_dir=args.cache_dir,
        match_all=args.match_all,
    )

    df = pd.read_csv(args.query_results_filepath, sep='\t', keep_default_na=False)
    df = df.sample(frac=0.4)
    generate_ph_pairs(
        df=df,
        ph_pairs_filepath=args.ph_pairs_filepath,
    )


if __name__ == "__main__":
    main()
