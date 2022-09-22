from ast import literal_eval
from collections import namedtuple, Counter
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
pd.options.mode.chained_assignment = None


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
        "species_with_part",
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
            type_: str,
            name: str,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ) -> pd.Series:
        # lower case
        type_ = type_.lower()
        name = name.lower()
        synonyms = [x.lower() for x in synonyms]

        # check integrity
        assert type_ in self._TYPES
        assert set(other_db_ids.keys()).issubset(self._OTHER_DBS)

        # check for duplicates
        def _check_duplicate(row):
            x = row["other_db_ids"]
            shared = {k: x[k] for k in x if k in other_db_ids and x[k] == other_db_ids[k]}
            return True if len(shared) >= 1 and row.type == type_ else False
        dup_idx = self.df_entities.apply(_check_duplicate, axis=1)

        df_duplicates = self.df_entities[dup_idx]

        # duplicates exist!
        if df_duplicates.shape[0] > 0:
            if df_duplicates.shape[0] >= 2:
                print(type_, name, synonyms, other_db_ids)
                print(df_duplicates)
                raise ValueError("Cannot have more than two matching rows!")

            entity = df_duplicates.iloc[0]
            name_and_synonyms = [entity.name] + entity.synonyms
            input_name_and_synonyms = [name] + synonyms

            new_synonyms = entity.synonyms
            for x in input_name_and_synonyms:
                if x in name_and_synonyms:
                    continue
                new_synonyms.append(x)

            entity.at["synonyms"] = new_synonyms
            entity.at["other_db_ids"] = {**other_db_ids, **entity.other_db_ids}

            self.df_entities.update(entity)
            return entity

        # no duplicates
        new_data = {
            "foodatlas_id": self.avail_id,
            "type": type_,
            "name": name,
            "synonyms": [x for x in synonyms],
            "other_db_ids": other_db_ids,
        }
        row = pd.Series(new_data, name=name)
        self.df_entities = pd.concat([self.df_entities, pd.DataFrame(row).transpose()])
        self.avail_id += 1
        return row

    def get_all_entities(self) -> pd.DataFrame:
        return self.df_entities

    def get_entity_by_id(self, foodatlas_id: int) -> pd.Series:
        raise NotImplementedError()

    def get_entity_by_name(self, name: str) -> pd.Series:
        raise NotImplementedError()

    def get_entities_by_type(self, type_: str) -> pd.DataFrame:
        return self.df_entities[self.df_entities["type"] == type_]

    def print_all_entities(self) -> None:
        print(self.df_entities)

    def num_entities(self) -> int:
        return self.df_entities.shape[0]

    def save(self, entities_filepath: str = None) -> None:
        if entities_filepath:
            print(f"Saving entities to a new filepath: {entities_filepath}")
            self.df_entities.to_csv(entities_filepath, sep='\t', index=False)
        else:
            print(f"Saving entities to original filepath: {self.entities_filepath}")
            self.df_entities.to_csv(self.entities_filepath, sep='\t', index=False)


class FoodAtlasRelation():
    _COLUMNS = [
        "foodatlas_id",
        "name",
        "translation",
    ]

    _DEFAULTS = [
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
    ) -> pd.Series:
        # check for duplicates
        for idx, row in self.df_relations.iterrows():
            if row.name == name:
                assert translation == row.translation
                return row

        # no duplicates
        new_data = {
            "foodatlas_id": self.avail_id,
            "name": name,
            "translation": translation,
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

    def save(self, relations_filepath: str = None) -> None:
        if relations_filepath:
            print(f"Saving relations to a new filepath: {relations_filepath}")
            self.df_relations.to_csv(relations_filepath, sep='\t', index=False)
        else:
            print(f"Saving relations to original filepath: {self.relations_filepath}")
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


def merge_candidate_entities(
    candidate_entities: List[CandidateEntity],
    using: str,
) -> List[CandidateEntity]:

    if using in ["NCBI_taxonomy", "MESH"]:
        duplicate_ids = [e.other_db_ids[using]
                         for e in candidate_entities if e.other_db_ids[using]]
        duplicate_ids = [x for x, count in Counter(duplicate_ids).items() if count > 1]

        if duplicate_ids:
            for duplicate_id in duplicate_ids:
                duplicates = [e for e in candidate_entities
                              if e.other_db_ids[using] == duplicate_id]

                type = []
                name = []
                synonyms = []
                other_db_ids = {}
                for d in duplicates:
                    if d.foodatlas_id is not None:
                        raise ValueError("Candidate entities cannot have foodatlas ID!")
                    type.append(d.type)
                    name.append(d.name)
                    synonyms.extend(d.synonyms)
                    other_db_ids = {**other_db_ids, **d.other_db_ids}

                type = list(set(type))
                name = list(set(name))
                synonyms = list(set(synonyms))

                assert len(type) == 1

                if type == "NCBI_taxonomy":
                    assert len(name) == 1
                elif type == "MESH":
                    if len(name) > 1:
                        synonyms = list(set(synonyms + name[1:]))
                        name = name[0]

                merged = CandidateEntity(
                    type=type[0],
                    name=name[0],
                    synonyms=synonyms,
                    other_db_ids=other_db_ids,
                )

                for d in duplicates:
                    candidate_entities.remove(d)
                candidate_entities.append(merged)

        return candidate_entities
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    fa_entity = FoodAtlasEntity("/home/jasonyoun/Jason/Research/FoodAtlas/data/FoodAtlas/entities.txt")

    temp = fa_entity.add(
        type="chemical",
        name="epicathechin",
        synonyms=[],
        other_db_ids={}
    )

    # fa_entity.save()
