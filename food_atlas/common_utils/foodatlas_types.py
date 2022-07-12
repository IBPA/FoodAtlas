from ast import literal_eval
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


class FoodAtlasEntity():
    _COLUMNS = [
        "foodatlas_id",
        "type",
        "name",
        "synonyms",
        "other_db_ids",
    ]

    _DEFAULTS = [
        None,
        None,
        None,
        [],
        {},
    ]

    _TYPES = [
        "chemical",
        "species",
        "species_with_part"
        "food_part",
    ]

    _OTHER_DBS = [
        "NCBI_taxonomy",
        "FooDB",
        "MESH",
    ]

    _CONVERTERS = {
        "synonyms": literal_eval,
        "other_db_ids": literal_eval,
    }

    def __init__(self, entities_filepath: str):
        self.entities_filepath = entities_filepath
        self.df_entities, self.avail_id = self._load()

    def _load(self):
        # if no entities file exists
        if not Path(self.entities_filepath).is_file():
            print("Entities file does not exist.")
            df_entities = pd.DataFrame(columns=self._COLUMNS)
            df_entities["foodatlas_id"] = df_entities["foodatlas_id"].astype(int)

            return df_entities, 0

        # if entities file exists
        print(f"Loading entities file from {self.entities_filepath}...")
        df_entities = pd.read_csv(self.entities_filepath, sep="\t", converters=self._CONVERTERS)
        df_entities["foodatlas_id"] = df_entities["foodatlas_id"].astype(int)
        df_entities.index = df_entities["name"].tolist()

        foodatlas_ids = df_entities["foodatlas_id"].tolist()
        avail_id = 0 if len(foodatlas_ids) == 0 else max(foodatlas_ids) + 1

        # check integrity
        types = df_entities.type.tolist()
        assert set(types).issubset(self._TYPES)

        other_dbs = [y for x in df_entities.other_db_ids.tolist() for y in x]
        assert set(other_dbs).issubset(self._OTHER_DBS)

        return df_entities, avail_id

    def add(
            self,
            type: str,
            name: str,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ) -> pd.Series:
        # check integrity
        assert type in self._TYPES
        assert set(other_db_ids.keys()).issubset(self._OTHER_DBS)

        # check for duplicates
        for idx, row in self.df_entities.iterrows():
            ns = [row.name] + row.synonyms
            input_ns = [name] + synonyms

            ns_lower = [x.lower() for x in ns]
            input_ns_lower = [x.lower() for x in input_ns]

            if set(ns_lower) & set(input_ns_lower) and row.type == type:
                print(f"Entity {name} already exists!")

                new_synonyms = row.synonyms
                for x in input_ns:
                    if x.lower() == row.name.lower() or x.lower() in ns_lower:
                        continue
                    new_synonyms.append(x)

                row.synonyms = new_synonyms

                # db_ids
                if len(other_db_ids) > 0:
                    input_db_ids = other_db_ids
                    db_ids = row.other_db_ids

                    if not set(input_db_ids.keys()) & set(db_ids.keys()):
                        db_ids = {**input_db_ids, **db_ids}
                    else:
                        # see if there is conflict
                        conflicting_dbs = list(set(input_db_ids.keys()) & set(db_ids.keys()))
                        for db in conflicting_dbs:
                            if input_db_ids[db] != db_ids[db]:
                                raise ValueError(f"DB ID does not match: {input_db_ids} | {db_ids}")

                    row.other_db_ids = db_ids

                self.df_entities.loc[idx] = row
                return row

        # no duplicates
        new_data = {
            "foodatlas_id": self.avail_id,
            "type": type,
            "name": name,
            "synonyms": synonyms,
            "other_db_ids": other_db_ids,
        }
        row = pd.Series(new_data, name=name)
        self.df_entities = pd.concat([self.df_entities, pd.DataFrame(row).transpose()])
        self.avail_id += 1
        return row

    def get_all_entities(self) -> pd.DataFrame:
        return self.df_entities

    def print_all_entities(self) -> None:
        print(self.df_entities)

    def num_entities(self) -> int:
        return self.df_entities.shape[0]

    def save(self) -> None:
        self.df_entities.to_csv(self.entities_filepath, sep='\t', index=False)


class FoodAtlasRelation():
    _COLUMNS = [
        "foodatlas_id",
        "name",
        "translation",
        "head_type",
        "tail_type",
    ]

    _DEFAULTS = [
        None,
        None,
        None,
        None,
        None,
    ]

    def __init__(self, relations_filepath: str):
        self.relations_filepath = relations_filepath
        self.df_relations, self.avail_id = self._load()

    def _load(self):
        # if no relations file exists
        if not Path(self.relations_filepath).is_file():
            print("Relations file does not exist.")
            df_relations = pd.DataFrame(columns=self._COLUMNS)
            df_relations["foodatlas_id"] = df_relations["foodatlas_id"].astype(int)

            return df_relations, 0

        # if relations file exists
        print(f"Loading relations file from {self.relations_filepath}...")
        df_relations = pd.read_csv(self.relations_filepath, sep="\t")
        df_relations["foodatlas_id"] = df_relations["foodatlas_id"].astype(int)
        df_relations.index = df_relations["name"].tolist()

        foodatlas_ids = df_relations["foodatlas_id"].tolist()
        avail_id = 0 if len(foodatlas_ids) == 0 else max(foodatlas_ids) + 1

        return df_relations, avail_id

    def add(
            self,
            name: str,
            translation: str,
            head_type: str,
            tail_type: str,
    ) -> pd.Series:
        # check for duplicates
        for idx, row in self.df_relations.iterrows():
            if row.name == name:
                assert translation == row.translation
                assert head_type == row.head_type
                assert tail_type == row.tail_type

                return row

        # no duplicates
        new_data = {
            "foodatlas_id": self.avail_id,
            "name": name,
            "translation": translation,
            "head_type": head_type,
            "tail_type": tail_type,
        }
        row = pd.Series(new_data, name=name)
        self.df_relations = pd.concat([self.df_relations, pd.DataFrame(row).transpose()])
        self.avail_id += 1
        return row

    def get_all_relations(self) -> pd.DataFrame:
        return self.df_relations

    def print_all_relations(self) -> None:
        print(self.df_relations)

    def num_relations(self) -> int:
        return self.df_relations.shape[0]

    def save(self) -> None:
        self.df_relations.to_csv(self.relations_filepath, sep='\t', index=False)


CandidateEntity = namedtuple(
    "CandidateEntity",
    FoodAtlasEntity._COLUMNS,
    defaults=FoodAtlasEntity._DEFAULTS,
)


CandidateRelation = namedtuple(
    "CandidateRelation",
    FoodAtlasRelation._COLUMNS,
    defaults=FoodAtlasRelation._DEFAULTS,
)


if __name__ == '__main__':
    fa_entity = FoodAtlasEntity("/home/jasonyoun/Jason/Research/FoodAtlas/data/FoodAtlas/entities.txt")

    temp = fa_entity.add(
        type="chemical",
        name="epicathechin",
        synonyms=[],
        other_db_ids={}
    )

    # fa_entity.save()
