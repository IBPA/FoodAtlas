from ast import literal_eval
from collections import namedtuple, OrderedDict
from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

from tqdm import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None


_KG_COLUMNS = [
    "head",
    "relation",
    "tail",
]

_EVIDENCE_COLUMNS = [
    "head",
    "relation",
    "tail",
    "triple",
    "pmid",
    "pmcid",
    "section",
    "premise",
    "source",
    "prob",
]

_ENTITY_COLUMNS = [
    "foodatlas_id",
    "type",
    "name",
    "synonyms",
    "other_db_ids",
]

_ENTITY_DEFAULTS = [
    None,
    None,
    None,
    [],
    {},
]

_ENTITY_TYPES = [
    # chemicl entity types
    "chemical",
    # organism entity types
    "organism",
    "organism:subgenus",
    "organism:subkingdom",
    "organism:subphylum",
    "organism:forma",
    "organism:tribe",
    "organism:parvorder",
    "organism:varietas",
    "organism:subspecies",
    "organism:subclass",
    "organism:forma specialis",
    "organism:superclass",
    "organism:species",
    "organism:infraorder",
    "organism:superorder",
    "organism:infraclass",
    "organism:superfamily",
    "organism:morph",
    "organism:genotype",
    "organism:serogroup",
    "organism:subtribe",
    "organism:serotype",
    "organism:genus",
    "organism:kingdom",
    "organism:species subgroup",
    "organism:no rank",
    "organism:strain",
    "organism:class",
    "organism:section",
    "organism:clade",
    "organism:superkingdom",
    "organism:family",
    "organism:species group",
    "organism:subsection",
    "organism:suborder",
    "organism:order",
    "organism:isolate",
    "organism:biotype",
    "organism:pathogroup",
    "organism:subfamily",
    "organism:superphylum",
    "organism:phylum",
    "organism:series",
    "organism:cohort",
    "organism:subcohort",
    # organism with part entity types
    "organism_with_part",
    "organism_with_part:subgenus",
    "organism_with_part:subkingdom",
    "organism_with_part:subphylum",
    "organism_with_part:forma",
    "organism_with_part:tribe",
    "organism_with_part:parvorder",
    "organism_with_part:varietas",
    "organism_with_part:subspecies",
    "organism_with_part:subclass",
    "organism_with_part:forma specialis",
    "organism_with_part:superclass",
    "organism_with_part:species",
    "organism_with_part:infraorder",
    "organism_with_part:superorder",
    "organism_with_part:infraclass",
    "organism_with_part:superfamily",
    "organism_with_part:morph",
    "organism_with_part:genotype",
    "organism_with_part:serogroup",
    "organism_with_part:subtribe",
    "organism_with_part:serotype",
    "organism_with_part:genus",
    "organism_with_part:kingdom",
    "organism_with_part:species subgroup",
    "organism_with_part:no rank",
    "organism_with_part:strain",
    "organism_with_part:class",
    "organism_with_part:section",
    "organism_with_part:clade",
    "organism_with_part:superkingdom",
    "organism_with_part:family",
    "organism_with_part:species group",
    "organism_with_part:subsection",
    "organism_with_part:suborder",
    "organism_with_part:order",
    "organism_with_part:isolate",
    "organism_with_part:biotype",
    "organism_with_part:pathogroup",
    "organism_with_part:subfamily",
    "organism_with_part:superphylum",
    "organism_with_part:phylum",
    "organism_with_part:series",
    "organism_with_part:cohort",
    "organism_with_part:subcohort",
]

_ENTITY_OTHER_DBS = [
    "NCBI_taxonomy",
    "FooDB",
    "MESH",
    "MESH_tree",
    "CAS",
]

_ENTITY_CONVERTERS = {
    "synonyms": literal_eval,
    "other_db_ids": literal_eval,
}

_RELATION_COLUMNS = [
    "foodatlas_id",
    "name",
    "translation",
]

_RELATION_DEFAULTS = [
    None,
    None,
    None,
]

CandidateEntity = namedtuple(
    "CandidateEntity",
    _ENTITY_COLUMNS,
    defaults=_ENTITY_DEFAULTS,
)

CandidateRelation = namedtuple(
    "CandidateRelation",
    _RELATION_COLUMNS,
    defaults=_RELATION_DEFAULTS,
)


class KnowledgeGraph():
    def __init__(
            self,
            kg_filepath: str,
            evidence_filepath: str,
            entities_filepath: str,
            relations_filepath: str,
    ):
        # load the graph
        self.kg_filepath = kg_filepath
        self.df_kg = self._read_kg()

        self.evidence_filepath = evidence_filepath
        self.df_evidence = self._read_evidence()

        # load the entities
        self.entities_filepath = entities_filepath
        self.df_entities, self.avail_entity_id = self._read_entities()

        # load the relations
        self.relations_filepath = relations_filepath
        self.df_relations, self.avail_relation_id = self._read_relations()

    def _read_kg(self) -> pd.DataFrame:
        # if no kg file exists
        if not Path(self.kg_filepath).is_file():
            print("KG file does not exist.")
            df_kg = pd.DataFrame(columns=_KG_COLUMNS)
            return df_kg

        # if entities file exists
        print(f"Loading KG file from {self.kg_filepath}...")
        df_kg = pd.read_csv(self.kg_filepath, sep='\t', keep_default_na=False)
        assert set(df_kg.columns.tolist()) == set(_KG_COLUMNS)
        df_kg = df_kg[_KG_COLUMNS]

        return df_kg

    def _read_evidence(self) -> pd.DataFrame:
        # if no evidence file exists
        if not Path(self.evidence_filepath).is_file():
            print("Evidence file does not exist.")
            df_evidence = pd.DataFrame(columns=_EVIDENCE_COLUMNS)
            return df_evidence

        # if entities file exists
        print(f"Loading evidence file from {self.evidence_filepath}...")
        df_evidence = pd.read_csv(
            self.evidence_filepath,
            sep='\t',
            keep_default_na=False,
            dtype={x: str for x in _EVIDENCE_COLUMNS}
        )
        assert set(df_evidence.columns.tolist()) == set(_EVIDENCE_COLUMNS)
        df_evidence = df_evidence[_EVIDENCE_COLUMNS]

        return df_evidence

    @staticmethod
    def _find_avail_foodatlas_id(ids: List[str]):
        prefix_str = list(set([x[0] for x in ids]))
        assert len(prefix_str) == 1
        prefix_str = prefix_str[0]
        ids = [int(x[1:]) for x in ids]
        return f"{prefix_str}{max(ids) + 1}"

    def _get_and_update_foodatlas_id(self, prefix_str: str):
        if prefix_str == "e":
            cur_id = self.avail_entity_id
            self.avail_entity_id = f"{prefix_str}{int(self.avail_entity_id[1:]) + 1}"
        elif prefix_str == "r":
            cur_id = self.avail_relation_id
            self.avail_relation_id = f"{prefix_str}{int(self.avail_relation_id[1:]) + 1}"
        else:
            raise ValueError()

        return cur_id

    def _read_entities(self):
        # if no entities file exists
        if not Path(self.entities_filepath).is_file():
            print("Entities file does not exist.")
            df_entities = pd.DataFrame(columns=_ENTITY_COLUMNS)
            return df_entities, "e0"

        # if entities file exists
        print(f"Loading entities file from {self.entities_filepath}...")
        df_entities = pd.read_csv(
            self.entities_filepath, sep="\t", converters=_ENTITY_CONVERTERS)

        avail_entity_id = KnowledgeGraph._find_avail_foodatlas_id(
            df_entities["foodatlas_id"].tolist())

        # check integrity
        types = df_entities.type.tolist()
        assert set(types).issubset(_ENTITY_TYPES)

        other_dbs = [y for x in df_entities.other_db_ids.tolist() for y in x]
        assert set(other_dbs).issubset(_ENTITY_OTHER_DBS)

        return df_entities, avail_entity_id

    def _read_relations(self):
        # if no relations file exists
        if not Path(self.relations_filepath).is_file():
            print("Relations file does not exist.")
            df_relations = pd.DataFrame(columns=_RELATION_COLUMNS)
            return df_relations, "r0"

        # if relations file exists
        print(f"Loading relations file from {self.relations_filepath}...")
        df_relations = pd.read_csv(self.relations_filepath, sep="\t")

        avail_relation_id = KnowledgeGraph._find_avail_foodatlas_id(
            df_relations["foodatlas_id"].tolist())

        return df_relations, avail_relation_id

    def add_ph_pairs(
            self,
            df: pd.DataFrame,
    ):
        assert set(["head", "relation", "tail", "premise"]).issubset(df.columns)

        entities = df["head"].tolist() + df["tail"].tolist()
        self._add_candidate_entities(entities)

        relations = df["relation"].tolist()
        self._add_candidate_relations(relations)

        print("Adding triples...")
        data = []
        evidence = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            head_foodatlas_id = self._find_matching_foodatlas_id(row["head"])
            relation_foodatlas_id = self._find_matching_foodatlas_id(row["relation"])
            tail_foodatlas_id = self._find_matching_foodatlas_id(row["tail"])

            triple = pd.Series({
                "head": head_foodatlas_id,
                "relation": relation_foodatlas_id,
                "tail": tail_foodatlas_id,
            })
            data.append(triple)

            e = pd.Series({
                "head": head_foodatlas_id,
                "relation": relation_foodatlas_id,
                "tail": tail_foodatlas_id,
                "triple": f"({head_foodatlas_id},{relation_foodatlas_id},{tail_foodatlas_id})",
                "pmid": row["pmid"],
                "pmcid": row["pmcid"],
                "section": row["section"],
                "premise": row["premise"],
                "source": row["source"],
                "prob": row["prob"] if row["source"].startswith("prediction") else "",
            })
            evidence.append(e)

            # has part
            if row["head"].type == "organism_with_part" or \
               row["head"].type.startswith("organism_with_part:"):
                relation = self._add_relation(
                    name="hasPart",
                    translation="has part",
                )

                head_without_part = deepcopy(row["head"])
                head_without_part = head_without_part._replace(
                    type=head_without_part.type.replace("organism_with_part", "organism"),
                    name=head_without_part.name.split(" - ")[0],
                    synonyms=[x.split(" - ")[0] for x in head_without_part.synonyms],
                )

                head_foodatlas_id = self._find_matching_foodatlas_id(head_without_part)
                tail_foodatlas_id = self._find_matching_foodatlas_id(row["head"])

                triple = pd.Series({
                    "head": head_foodatlas_id,
                    "relation": relation.foodatlas_id,
                    "tail": tail_foodatlas_id,
                })
                data.append(triple)

                e = pd.Series({
                    "head": head_foodatlas_id,
                    "relation": relation.foodatlas_id,
                    "tail": tail_foodatlas_id,
                    "triple": f"({head_foodatlas_id},{relation.foodatlas_id},{tail_foodatlas_id})",
                    "pmid": row["pmid"],
                    "pmcid": row["pmcid"],
                    "section": row["section"],
                    "premise": row["premise"],
                    "source": row["source"],
                    "prob": row["prob"] if row["source"].startswith("prediction") else "",
                })
                evidence.append(e)

        self.df_kg = pd.concat([self.df_kg, pd.DataFrame(data, columns=_KG_COLUMNS)])
        self.df_kg.drop_duplicates(inplace=True, ignore_index=True)

        self.df_evidence = pd.concat(
            [self.df_evidence, pd.DataFrame(evidence, columns=_EVIDENCE_COLUMNS)])
        self.df_evidence.drop_duplicates(inplace=True, ignore_index=True)

        return self.df_kg, self.df_evidence

    def add_triples(
            self,
            df: pd.DataFrame,
            origin: str,
    ):
        assert origin in ["NCBI_taxonomy", "MESH"]
        assert set(["head", "relation", "tail"]).issubset(df.columns)

        entities = df["head"].tolist() + df["tail"].tolist()
        self._add_candidate_entities(entities)

        relations = df["relation"].tolist()
        self._add_candidate_relations(relations)

        print("Adding triples...")
        data = []
        evidence = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            head_foodatlas_id = self._find_matching_foodatlas_id(row["head"])
            relation_foodatlas_id = self._find_matching_foodatlas_id(row["relation"])
            tail_foodatlas_id = self._find_matching_foodatlas_id(row["tail"])

            triple = pd.Series({
                "head": head_foodatlas_id,
                "relation": relation_foodatlas_id,
                "tail": tail_foodatlas_id,
            })
            data.append(triple)

            e = pd.Series({
                "head": head_foodatlas_id,
                "relation": relation_foodatlas_id,
                "tail": tail_foodatlas_id,
                "triple": f"({head_foodatlas_id},{relation_foodatlas_id},{tail_foodatlas_id})",
                "pmid": None,
                "pmcid": None,
                "section": None,
                "premise": None,
                "source": origin,
                "prob": None,
            })
            evidence.append(e)

        self.df_kg = pd.concat([self.df_kg, pd.DataFrame(data, columns=_KG_COLUMNS)])
        self.df_kg.drop_duplicates(inplace=True, ignore_index=True)

        self.df_evidence = pd.concat(
            [self.df_evidence, pd.DataFrame(evidence, columns=_EVIDENCE_COLUMNS)])
        self.df_evidence.drop_duplicates(inplace=True, ignore_index=True)

        return self.df_kg, self.df_evidence

    def _add_candidate_entities(
            self,
            entities: List[CandidateEntity],
    ) -> pd.Series:
        print(f"Number of entities before updating: {self.df_entities.shape[0]}")

        print("Preparing to add entities...")
        entities = [eval(e, globals()) for e in sorted(set([str(e) for e in entities]))]
        data = [["", e.type, e.name, e.synonyms, e.other_db_ids] for e in entities]
        df_entities_to_add = pd.DataFrame(data, columns=_ENTITY_COLUMNS)

        # extract organisms from organisms_with_part
        df_extracted_organisms = \
            df_entities_to_add[df_entities_to_add["type"].apply(
                lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]
        df_extracted_organisms["type"] = df_extracted_organisms["type"].apply(
            lambda x: x.replace("organism_with_part", "organism"))
        df_extracted_organisms["name"] = df_extracted_organisms["name"].apply(
            lambda x: x.split(" - ")[0])
        df_extracted_organisms["synonyms"] = df_extracted_organisms["synonyms"].apply(
            lambda synonyms: [x.split(" - ")[0] for x in synonyms])

        df_entities = pd.concat([
            self.df_entities.copy(), df_entities_to_add, df_extracted_organisms])
        df_entities.reset_index(inplace=True, drop=True)

        duplicate_idx = []

        # organisms
        print("Finding duplicate organisms...")
        df_organisms = df_entities[df_entities["type"].apply(
            lambda x: x == "organism" or x.startswith("organism:"))]

        for db in ["NCBI_taxonomy"]:
            df_organisms[db] = df_organisms["other_db_ids"].apply(lambda x: x[db])
            for db_id, df_subset in df_organisms.groupby(db):
                if df_subset.shape[0] == 1:
                    continue
                duplicate_idx.append(list(df_subset.index))

        # organisms with part
        print("Finding duplicate organisms with part...")
        df_organisms_with_part = df_entities[df_entities["type"].apply(
            lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]

        for db in ["NCBI_taxonomy"]:
            df_organisms_with_part[db] = df_organisms_with_part.apply(
                lambda row: row["other_db_ids"][db]+row["name"].split(" - ")[1], axis=1)
            for db_id, df_subset in df_organisms_with_part.groupby(db):
                if df_subset.shape[0] == 1:
                    continue
                duplicate_idx.append(list(df_subset.index))

        # chemicals
        print("Finding duplicate chemicals...")
        df_chemicals = df_entities[df_entities["type"] == "chemical"]

        for db in ["MESH"]:
            df_chemicals[db] = df_chemicals["other_db_ids"].apply(lambda x: x[db])
            for db_id, df_subset in df_chemicals.groupby(db):
                if df_subset.shape[0] == 1:
                    continue
                duplicate_idx.append(list(df_subset.index))

        # merge duplicate idx
        print("Merging duplicate idexes...")
        flattened_idx = list(set([y for x in duplicate_idx for y in x]))

        for x in flattened_idx:
            match = [j for j in duplicate_idx if x in j]
            if len(match) == 1:
                continue
            match_merged = list(set([k for j in match for k in j]))
            no_match = [j for j in duplicate_idx if x not in j]
            duplicate_idx = [match_merged] + no_match

        # merge duplicates
        print("Merging duplicate entities...")
        merged_duplicates = []
        for idx in duplicate_idx:
            df_duplicates = df_entities.iloc[idx]
            assert len(set(df_duplicates["type"].tolist())) == 1
            _type = df_duplicates["type"].tolist()[0]

            ids = list(set(df_duplicates["foodatlas_id"].tolist()))
            if "" in ids:
                ids.remove("")

            if len(ids) == 0:
                foodatlas_id = ""
                name = df_duplicates.iloc[0]["name"]
            elif len(ids) == 1:
                foodatlas_id = ids[0]
                name = df_duplicates[df_duplicates["foodatlas_id"] != ""].iloc[0]["name"]
            else:
                raise ValueError("Cannot have more than two ids")

            names = df_duplicates["name"].tolist()
            synonyms = [x for subset in df_duplicates["synonyms"].tolist() for x in subset]
            to_merge = list(set(names + synonyms))
            to_merge.remove(name)
            synonyms = KnowledgeGraph._merge_synonyms(to_merge)
            if name.lower() in synonyms:
                synonyms.remove(name.lower())

            other_db_ids = {}
            for d in df_duplicates["other_db_ids"].tolist():
                other_db_ids.update(d)

            merged_duplicates.append([foodatlas_id, _type, name, synonyms, other_db_ids])

        flattened_idx = list(set([y for x in duplicate_idx for y in x]))
        non_duplicate_idx = [x for x in df_entities.index if x not in flattened_idx]

        df_entities = pd.concat([
            df_entities.iloc[non_duplicate_idx],
            pd.DataFrame(merged_duplicates, columns=_ENTITY_COLUMNS),
        ])

        df_entities_with_id = df_entities[df_entities["foodatlas_id"] != ""]
        df_entities_no_id = df_entities[df_entities["foodatlas_id"] == ""]
        df_entities_no_id["foodatlas_id"] = df_entities_no_id.apply(
            lambda _: self._get_and_update_foodatlas_id(prefix_str="e"), axis=1)

        self.df_entities = pd.concat([df_entities_with_id, df_entities_no_id])
        self.df_entities.reset_index(inplace=True, drop=True)

        print(f"Number of entities after updating: {self.df_entities.shape[0]}")

    def _add_candidate_relations(
            self,
            relations: List[CandidateRelation],
    ) -> pd.Series:
        print(f"Number of relations before updating: {self.df_relations.shape[0]}")

        relations = [eval(r, globals()) for r in sorted(set([str(r) for r in relations]))]
        data = [["", r.name, r.translation] for r in relations]
        df_relations = pd.concat([
            self.df_relations.copy(),
            pd.DataFrame(data, columns=_RELATION_COLUMNS),
        ])
        df_relations.drop_duplicates("name", inplace=True)
        df_relations_with_id = df_relations[df_relations["foodatlas_id"] != ""]
        df_relations_no_id = df_relations[df_relations["foodatlas_id"] == ""]
        df_relations_no_id["foodatlas_id"] = df_relations_no_id.apply(
            lambda _: self._get_and_update_foodatlas_id(prefix_str="r"), axis=1)

        self.df_relations = pd.concat([df_relations_with_id, df_relations_no_id])
        self.df_relations.reset_index(inplace=True, drop=True)

        print(f"Number of relations after updating: {self.df_relations.shape[0]}")

    def _find_matching_foodatlas_id(self, entity_or_relation):
        if type(entity_or_relation) == CandidateEntity:
            df_match = self.df_entities[self.df_entities["other_db_ids"].apply(
                lambda x: x.items() >= entity_or_relation.other_db_ids.items())]
            df_match = df_match[df_match["type"] == entity_or_relation.type]

            if entity_or_relation.type == "organism_with_part":
                df_match = df_match[df_match["name"].apply(
                    lambda x: x.split(" - ")[1] == entity_or_relation.name.split(" - ")[1])]

            assert df_match.shape[0] == 1
            return df_match.iloc[0]["foodatlas_id"]
        elif type(entity_or_relation) == CandidateRelation:
            df_match = self.df_relations[self.df_relations["name"] == entity_or_relation.name]
            assert df_match.shape[0] == 1
            return df_match.iloc[0]["foodatlas_id"]
        else:
            raise ValueError()

    @staticmethod
    def _merge_synonyms(synonyms):
        all_synonyms = set(synonyms)
        seen = OrderedDict()
        for word in all_synonyms:
            lo = word.lower()
            seen[lo] = min(word, seen.get(lo, word))
        return list(seen.values())

    def _update_entity(
            self,
            foodatlas_id: str,
            type_: str = None,
            name: str = None,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ):
        self.df_entities.set_index("foodatlas_id", drop=False, inplace=True)

        if type_:
            self.df_entities.at[foodatlas_id, "type"] = type_
            if type_.startswith("organism_with_part"):
                part_name = self.df_entities.at[foodatlas_id, "name"].split(" - ")[-1]
        if name:
            all_names = self.df_entities.at[foodatlas_id, "synonyms"].copy()
            all_names += [self.df_entities.at[foodatlas_id, "name"]]
            all_names += [name]
            all_names = list(set(all_names))

            if name in all_names:
                all_names.remove(name)
            if name.lower() in all_names:
                all_names.remove(name.lower())

            if type_.startswith("organism_with_part"):
                self.df_entities.at[foodatlas_id, "name"] = f"{name} - {part_name}"
                self.df_entities.at[foodatlas_id, "synonyms"] = \
                    KnowledgeGraph._merge_synonyms(all_names)
            else:
                self.df_entities.at[foodatlas_id, "name"] = name
                self.df_entities.at[foodatlas_id, "synonyms"] = \
                    KnowledgeGraph._merge_synonyms(all_names)

        if synonyms:
            all_names = self.df_entities.at[foodatlas_id, "synonyms"].copy()
            if type_.startswith("organism_with_part"):
                all_names += [f"{x} - {part_name}" for x in synonyms]
            else:
                all_names += synonyms
            all_names = set(all_names)

            if self.df_entities.at[foodatlas_id, "name"] in all_names:
                all_names.remove(self.df_entities.at[foodatlas_id, "name"])

            if self.df_entities.at[foodatlas_id, "name"].lower() in all_names:
                all_names.remove(self.df_entities.at[foodatlas_id, "name"].lower())

            self.df_entities.at[foodatlas_id, "synonyms"] = \
                KnowledgeGraph._merge_synonyms(all_names)

        if other_db_ids:
            self.df_entities.at[foodatlas_id, "other_db_ids"] = other_db_ids

        return self.df_entities.loc[foodatlas_id]

    def get_kg(self) -> pd.DataFrame:
        return self.df_kg.copy()

    def get_evidence(self) -> pd.DataFrame:
        return self.df_evidence.copy()

    def get_all_entities(self) -> pd.DataFrame:
        return self.df_entities.copy()

    def get_entity_by_id(self, foodatlas_id: str) -> pd.Series:
        entity = self.df_entities[self.df_entities["foodatlas_id"] == foodatlas_id]
        assert entity.shape[0] == 1
        return entity.iloc[0].copy()

    def get_entity_by_other_db_id(self, db_name: str, db_id: str) -> pd.Series:
        entity = self.df_entities[self.df_entities["other_db_ids"].apply(lambda x: db_name in x)]

        def _match_db_id(x, y):
            if type(x) == str:
                return x == y
            elif type(x) == list:
                return y in x
            else:
                raise ValueError

        entity = entity[entity["other_db_ids"].apply(lambda x: _match_db_id(x[db_name], db_id))]
        assert entity.shape[0] == 1
        return entity.iloc[0].copy()

    def get_entity_by_name(self, name: str) -> pd.Series:
        raise NotImplementedError()

    def get_entities_by_type(
            self,
            type_: str = None,
            startswith: str = None,
    ) -> pd.DataFrame:
        if type_:
            return self.df_entities[self.df_entities["type"] == type_].copy()
        if startswith:
            return self.df_entities[self.df_entities["type"].startswith(startswith)].copy()

        raise RuntimeError()

    def print_all_entities(self) -> None:
        print(self.df_entities)

    def num_entities(self) -> int:
        return self.df_entities.shape[0]

    def _add_relation(
            self,
            name: str,
            translation: str,
    ) -> pd.Series:
        # check for duplicates
        for idx, row in self.df_relations.iterrows():
            if row["name"] == name:
                assert translation == row.translation
                return row

        # no duplicates
        new_data = {
            "foodatlas_id": self._get_and_update_foodatlas_id("r"),
            "name": name,
            "translation": translation,
        }
        row = pd.Series(new_data)
        self.df_relations = pd.concat([self.df_relations, pd.DataFrame(row).transpose()])
        self.df_relations.sort_values("foodatlas_id", inplace=True, ignore_index=True)

        return row

    def get_all_relations(self) -> pd.DataFrame:
        return self.df_relations.copy()

    def get_relation_by_id(self, foodatlas_id: str) -> pd.Series:
        relation = self.df_relations[self.df_relations["foodatlas_id"] == foodatlas_id]
        assert relation.shape[0] == 1
        return relation.iloc[0]

    def print_all_relations(self) -> None:
        print(self.df_relations)

    def num_relations(self) -> int:
        return self.df_relations.shape[0]

    def save(
            self,
            kg_filepath: str = None,
            evidence_filepath: str = None,
            entities_filepath: str = None,
            relations_filepath: str = None,
    ) -> None:
        self.df_entities["foodatlas_id_val_only"] = self.df_entities["foodatlas_id"].apply(
            lambda x: int(x[1:]))
        self.df_entities.sort_values("foodatlas_id_val_only", inplace=True, ignore_index=True)

        self.df_relations["foodatlas_id_val_only"] = self.df_relations["foodatlas_id"].apply(
            lambda x: int(x[1:]))
        self.df_relations.sort_values("foodatlas_id_val_only", inplace=True, ignore_index=True)

        self.df_entities.drop("foodatlas_id_val_only", inplace=True, axis=1)
        self.df_relations.drop("foodatlas_id_val_only", inplace=True, axis=1)

        if kg_filepath:
            Path(kg_filepath).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving kg to a new filepath: {kg_filepath}")
            self.df_kg.to_csv(kg_filepath, sep='\t', index=False)
        else:
            print(f"Saving kg to original filepath: {self.kg_filepath}")
            self.df_kg.to_csv(self.kg_filepath, sep='\t', index=False)

        if evidence_filepath:
            Path(evidence_filepath).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving evidence to a new filepath: {evidence_filepath}")
            self.df_evidence.to_csv(evidence_filepath, sep='\t', index=False)
        else:
            print(f"Saving evidence to original filepath: {self.evidence_filepath}")
            self.df_evidence.to_csv(self.evidence_filepath, sep='\t', index=False)

        if entities_filepath:
            Path(entities_filepath).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving entities to a new filepath: {entities_filepath}")
            self.df_entities.to_csv(entities_filepath, sep='\t', index=False)
        else:
            print(f"Saving entities to original filepath: {self.entities_filepath}")
            self.df_entities.to_csv(self.entities_filepath, sep='\t', index=False)

        if relations_filepath:
            Path(relations_filepath).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving relations to a new filepath: {relations_filepath}")
            self.df_relations.to_csv(relations_filepath, sep='\t', index=False)
        else:
            print(f"Saving relations to original filepath: {self.relations_filepath}")
            self.df_relations.to_csv(self.relations_filepath, sep='\t', index=False)


def main():
    df = pd.read_csv("../../data/toy/test_kg_api.txt", sep='\t', keep_default_na=False)
    # df = pd.read_csv("../../outputs/data_generation/train_pool.tsv", sep='\t', keep_default_na=False)
    df = df[df["answer"] == "Entails"]
    df["head"] = df["head"].apply(lambda x: eval(x, globals()))
    df["relation"] = df["relation"].apply(lambda x: eval(x, globals()))
    df["tail"] = df["tail"].apply(lambda x: eval(x, globals()))

    temporary_dir = "/home/jasonyoun/Jason/Scratch/temp"
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(temporary_dir, "kg.txt"),
        evidence_filepath=os.path.join(temporary_dir, "evidence.txt"),
        entities_filepath=os.path.join(temporary_dir, "entities.txt"),
        relations_filepath=os.path.join(temporary_dir, "relations.txt"),
    )
    fa_kg.add_ph_pairs(df)
    fa_kg.save()
    # fa_kg.save(
    #     kg_filepath=os.path.join(temporary_dir.replace("temp", "temp2"), "kg.txt"),
    #     evidence_filepath=os.path.join(temporary_dir.replace("temp", "temp2"), "evidence.txt"),
    #     entities_filepath=os.path.join(temporary_dir.replace("temp", "temp2"), "entities.txt"),
    #     relations_filepath=os.path.join(temporary_dir.replace("temp", "temp2"), "relations.txt"),
    # )


if __name__ == '__main__':
    main()
