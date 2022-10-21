from ast import literal_eval
from collections import namedtuple, Counter, OrderedDict
from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

from tqdm import tqdm
import numpy as np
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
    "chemical:superclass",
    "chemical:class",
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
    "ClassyFire",
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

    ######
    # KG #
    ######
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
        df_evidence = pd.read_csv(self.evidence_filepath, sep='\t', keep_default_na=False)
        assert set(df_evidence.columns.tolist()) == set(_EVIDENCE_COLUMNS)
        df_evidence["prob"] = df_evidence["prob"].apply(lambda x: float(x) if x != '' else None)
        df_evidence = df_evidence[_EVIDENCE_COLUMNS]

        return df_evidence

    def get_kg(self) -> pd.DataFrame:
        return self.df_kg.copy()

    def get_evidence(self) -> pd.DataFrame:
        return self.df_evidence.copy()

    def _and_entity_and_build_lookup(self, new_entities):
        print("Adding entities...")
        entity_lookup = {x: {} for x in _ENTITY_TYPES}
        for x in tqdm(new_entities):
            e = self._add_entity(
                type_=x.type,
                name=x.name,
                synonyms=x.synonyms,
                other_db_ids=x.other_db_ids,
            )

            key = KnowledgeGraph._get_other_db_id(x)
            if key not in entity_lookup[x.type]:
                entity_lookup[x.type][key] = e

        for _, row in tqdm(self.df_entities.iterrows(), total=self.df_entities.shape[0]):
            key = KnowledgeGraph._get_other_db_id(row)
            if key not in entity_lookup[row["type"]]:
                entity_lookup[row["type"]][key] = row

        return entity_lookup

    @staticmethod
    def _get_derivative_organisms(organisms_with_part):
        # extract organism from organism_with_part
        derivative_organisms = []
        for owp in organisms_with_part:
            organism = deepcopy(owp)
            organism = organism._replace(
                type="organism",
                name=owp.name.split(' - ')[0],
                synonyms=[x.split(' - ')[0] for x in owp.synonyms],
            )

            derivative_organisms.append(organism)

        return derivative_organisms

    def add_ph_pairs(
            self,
            df_input: pd.DataFrame,
    ):
        # add the entities first for speed
        entities = df_input["head"].tolist() + df_input["tail"].tolist()
        chemicals = [x for x in entities if x.type.startswith("chemical")]
        organisms = [x for x in entities
                     if x.type.startswith("organism") and
                     not x.type.startswith("organism_with_part")]
        organisms_with_part = [x for x in entities if x.type.startswith("organism_with_part")]
        assert len(entities) == (len(chemicals) + len(organisms) + len(organisms_with_part))
        organisms += KnowledgeGraph._get_derivative_organisms(organisms_with_part)

        print(f"Number of chemicals before merging: {len(chemicals)}")
        print(f"Number of organisms before merging: {len(organisms)}")
        print(f"Number of organisms_with_part before merging: {len(organisms_with_part)}")

        print("Merging chemicals...")
        chemicals = KnowledgeGraph.merge_candidate_entities(
            chemicals, candidates_type="chemical")
        print("Merging organisms...")
        organisms = KnowledgeGraph.merge_candidate_entities(
            organisms, candidates_type="organism")
        print("Merging organisms_with_part...")
        organisms_with_part = KnowledgeGraph.merge_candidate_entities(
            organisms_with_part, candidates_type="organism_with_part")
        entities = chemicals + organisms + organisms_with_part

        print(f"Number of chemicals after merging: {len(chemicals)}")
        print(f"Number of organisms after merging: {len(organisms)}")
        print(f"Number of organisms_with_part after merging: {len(organisms_with_part)}")

        entity_lookup = self._and_entity_and_build_lookup(entities)

        print("Adding triples...")
        data = []
        evidence = []
        for _, row in tqdm(df_input.iterrows(), total=df_input.shape[0]):
            head = row["head"]
            tail = row["tail"]

            relation = self._add_relation(
                name=row["relation"].name,
                translation=row["relation"].translation,
            )

            head_foodatlas_id = \
                entity_lookup[head.type][KnowledgeGraph._get_other_db_id(head)].foodatlas_id
            tail_foodatlas_id = \
                entity_lookup[tail.type][KnowledgeGraph._get_other_db_id(tail)].foodatlas_id

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
                "prob": float(row["prob"]) if row["source"].startswith("prediction") else None,
            })
            evidence.append(e)

            # has part
            if head.type == "organism_with_part":
                relation = self._add_relation(
                    name="hasPart",
                    translation="has part",
                )

                head_foodatlas_id = \
                    entity_lookup["organism"][KnowledgeGraph._get_other_db_id(head).split('-')[0]].foodatlas_id
                tail_foodatlas_id = \
                    entity_lookup[head.type][KnowledgeGraph._get_other_db_id(head)].foodatlas_id

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
                    "prob": float(row["prob"]) if row["source"].startswith("prediction") else None,
                })
                evidence.append(e)

        self.df_kg = pd.concat([self.df_kg, pd.DataFrame(data, columns=_KG_COLUMNS)])
        self.df_kg.drop_duplicates(inplace=True, ignore_index=True)

        self.df_evidence = pd.concat(
            [self.df_evidence, pd.DataFrame(evidence, columns=_EVIDENCE_COLUMNS)])

        def _merge_prob(probs):
            probs = [x for x in list(probs) if x is not None]
            if len(probs) == 0:
                return None
            else:
                return np.mean(probs)
        evidence_columns_without_prob = [x for x in _EVIDENCE_COLUMNS if x != "prob"]
        self.df_evidence = self.df_evidence \
            .groupby(evidence_columns_without_prob)["prob"] \
            .agg(lambda probs: _merge_prob(probs)).reset_index()

        self.df_evidence.drop_duplicates(inplace=True, ignore_index=True)

        return self.df_kg, self.df_evidence

    @staticmethod
    def _get_other_db_id(x):
        if type(x) == CandidateEntity:
            if x.type.startswith("chemical"):
                return x.other_db_ids["MESH"]
            elif x.type.split(":")[0] == "organism":
                return x.other_db_ids["NCBI_taxonomy"]
            elif x.type.split(":")[0] == "organism_with_part":
                part_name = x.name.split(' - ')[-1]
                return x.other_db_ids["NCBI_taxonomy"] + f"-{part_name}"
            else:
                raise NotImplementedError()
        elif type(x) == pd.Series:
            if x["type"].startswith("chemical"):
                return x["other_db_ids"]["MESH"]
            elif x["type"].split(":")[0] == "organism":
                return x["other_db_ids"]["NCBI_taxonomy"]
            elif x["type"].split(":")[0] == "organism_with_part":
                part_name = x["name"].split(' - ')[-1]
                return x["other_db_ids"]["NCBI_taxonomy"] + f"-{part_name}"
            else:
                raise NotImplementedError()

    def add_taxonomy(
            self,
            df_input: pd.DataFrame,
            origin: str,
    ):
        assert origin in ["NCBI_taxonomy", "ClassyFire"]

        # add the entities first for speed
        entities = df_input["head"].tolist() + df_input["tail"].tolist()
        chemicals = [x for x in entities if x.type.startswith("chemical")]
        organisms = [x for x in entities if x.type.startswith("organism")]
        assert len(entities) == (len(chemicals) + len(organisms))

        print(f"Number of chemicals before merging: {len(chemicals)}")
        print(f"Number of organisms before merging: {len(organisms)}")

        print("Merging chemicals...")
        chemicals = KnowledgeGraph.merge_candidate_entities(chemicals, candidates_type="chemical")
        print("Merging organisms...")
        organisms = KnowledgeGraph.merge_candidate_entities(organisms, candidates_type="organism")
        entities = chemicals + organisms

        print(f"Number of chemicals after merging: {len(chemicals)}")
        print(f"Number of organisms after merging: {len(organisms)}")

        entity_lookup = self._and_entity_and_build_lookup(entities)

        print("Adding triples...")
        data = []
        evidence = []
        for _, row in tqdm(df_input.iterrows(), total=df_input.shape[0]):
            head = row["head"]
            tail = row["tail"]

            # relation
            relation = self._add_relation(
                name=row["relation"].name,
                translation=row["relation"].translation,
            )

            head_foodatlas_id = \
                entity_lookup[head.type][KnowledgeGraph._get_other_db_id(head)].foodatlas_id
            tail_foodatlas_id = \
                entity_lookup[tail.type][KnowledgeGraph._get_other_db_id(tail)].foodatlas_id

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

    ############
    # ENTITIES #
    ############
    def _read_entities(self):
        # if no entities file exists
        if not Path(self.entities_filepath).is_file():
            print("Entities file does not exist.")
            df_entities = pd.DataFrame(columns=_ENTITY_COLUMNS)
            df_entities["foodatlas_id"] = df_entities["foodatlas_id"].astype(int)

            return df_entities, 0

        # if entities file exists
        print(f"Loading entities file from {self.entities_filepath}...")
        df_entities = pd.read_csv(
            self.entities_filepath, sep="\t", converters=_ENTITY_CONVERTERS)
        df_entities["foodatlas_id"] = df_entities["foodatlas_id"].astype(int)
        df_entities.index = df_entities["foodatlas_id"].tolist()

        foodatlas_ids = df_entities["foodatlas_id"].tolist()
        avail_entity_id = 0 if len(foodatlas_ids) == 0 else max(foodatlas_ids) + 1

        # check integrity
        types = df_entities.type.tolist()
        assert set(types).issubset(_ENTITY_TYPES)

        other_dbs = [y for x in df_entities.other_db_ids.tolist() for y in x]
        assert set(other_dbs).issubset(_ENTITY_OTHER_DBS)

        return df_entities, avail_entity_id

    def _add_entity(
            self,
            type_: str,
            name: str,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ) -> pd.Series:
        # check integrity
        assert type_ in _ENTITY_TYPES
        assert set(other_db_ids.keys()).issubset(_ENTITY_OTHER_DBS)

        if type_.startswith("chemical"):
            unique_id = other_db_ids["MESH"]
        elif type_.split(":")[0] == "organism":
            unique_id = other_db_ids["NCBI_taxonomy"]
        elif type_.split(":")[0] == "organism_with_part":
            part_name = name.split(' - ')[-1]
            unique_id = other_db_ids["NCBI_taxonomy"] + f"-{part_name}"

        # check for duplicates
        def _check_duplicate(row):
            return True if unique_id == KnowledgeGraph._get_other_db_id(row) else False
        dup_idx = self.df_entities.apply(_check_duplicate, axis=1)
        df_duplicates = self.df_entities[dup_idx]

        # duplicates exist!
        if df_duplicates.shape[0] > 0:
            if df_duplicates.shape[0] >= 2:
                print(type_, name, synonyms, other_db_ids)
                print(df_duplicates)
                raise ValueError("Cannot have more than two matching rows!")

            entity = df_duplicates.iloc[0]
            all_names = set([entity["name"]] + entity["synonyms"] + [name] + synonyms)
            if entity["name"] in all_names:
                all_names.remove(entity["name"])
            if entity["name"].lower() in all_names:
                all_names.remove(entity["name"].lower())

            entity.at["synonyms"] = KnowledgeGraph._merge_synonyms(all_names)
            entity.at["other_db_ids"] = {**other_db_ids, **entity["other_db_ids"]}

            self.df_entities.update(entity)
            return entity

        # no duplicates
        new_data = {
            "foodatlas_id": self.avail_entity_id,
            "type": type_,
            "name": name,
            "synonyms": KnowledgeGraph._merge_synonyms(synonyms),
            "other_db_ids": other_db_ids,
        }
        entity = pd.Series(new_data, name=name)
        self.df_entities = pd.concat([self.df_entities, pd.DataFrame(entity).transpose()])
        self.avail_entity_id += 1

        return entity

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
            foodatlas_id: int,
            type_: str = None,
            name: str = None,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ) -> pd.Series:
        if type_:
            self.df_entities.at[foodatlas_id, "type"] = type_

            if type_.startswith("organism_with_part"):
                part_name = self.df_entities.at[foodatlas_id, "name"].split(" - ")[-1]
        if name:
            all_names = self.df_entities.at[foodatlas_id, "synonyms"].copy()
            all_names += [self.df_entities.at[foodatlas_id, "name"]]

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
            raise NotImplementedError

        return self.df_entities.loc[foodatlas_id]

    @staticmethod
    def merge_candidate_entities(
        candidate_entities: List[CandidateEntity],
        candidates_type: str,
    ) -> List[CandidateEntity]:

        def _merge_duplicates(duplicates):
            type_ = []
            name = []
            synonyms = []
            other_db_ids = {}
            for d in duplicates:
                if d.foodatlas_id is not None:
                    raise ValueError("Candidate entities cannot have foodatlas ID!")
                type_.append(d.type)
                name.append(d.name)
                synonyms.extend(d.synonyms)
                other_db_ids = {**other_db_ids, **d.other_db_ids}

            type_ = list(set(type_))
            name = list(set(name))
            synonyms = list(set(synonyms))

            assert len(type_) == 1

            if type_ == "NCBI_taxonomy":
                assert len(name) == 1
            elif type_ == "MESH":
                if len(name) > 1:
                    synonyms = list(set(synonyms + name[1:]))
                    name = name[0]

            merged = CandidateEntity(
                type=type_[0],
                name=name[0],
                synonyms=synonyms,
                other_db_ids=other_db_ids,
            )

            for d in duplicates:
                candidate_entities.remove(d)
            candidate_entities.append(merged)

        if candidates_type.split(':')[0] in ["chemical", "organism"]:
            if candidates_type.split(':')[0] == "organism":
                using = "NCBI_taxonomy"
            elif candidates_type.split(':')[0] == "chemical":
                using = "MESH"
            else:
                raise ValueError()

            duplicate_ids = [e.other_db_ids[using]
                             for e in candidate_entities if e.other_db_ids[using]]
            duplicate_ids = [x for x, count in Counter(duplicate_ids).items() if count > 1]

            if duplicate_ids:
                for duplicate_id in tqdm(duplicate_ids):
                    duplicates = [e for e in candidate_entities
                                  if e.other_db_ids[using] == duplicate_id]
                    _merge_duplicates(duplicates)

            return candidate_entities
        elif candidates_type.split(':')[0] == "organism_with_part":
            using = "NCBI_taxonomy"
            unique_ids = {}
            for e in candidate_entities:
                key = e.other_db_ids[using] + e.name.split(' - ')[-1]
                if key in unique_ids:
                    unique_ids[key].append(e)
                else:
                    unique_ids[key] = [e]

            duplicate_ids = [k for k, v in unique_ids.items() if len(v) > 1]

            for k, v in unique_ids.items():
                if len(v) < 2:
                    continue
                _merge_duplicates(v)

            return candidate_entities
        else:
            raise NotImplementedError()

    def get_all_entities(self) -> pd.DataFrame:
        return self.df_entities

    def get_entity_by_id(self, foodatlas_id: int) -> pd.Series:
        entity = self.df_entities[self.df_entities["foodatlas_id"] == foodatlas_id]
        assert entity.shape[0] == 1
        return entity.iloc[0]

    def get_entity_by_name(self, name: str) -> pd.Series:
        raise NotImplementedError()

    def get_entities_by_type(
            self,
            type_: str = None,
            startswith: str = None,
    ) -> pd.DataFrame:
        if type_:
            return self.df_entities[self.df_entities["type"] == type_]
        if startswith:
            return self.df_entities[self.df_entities["type"].startswith(startswith)]

        raise RuntimeError()

    def print_all_entities(self) -> None:
        print(self.df_entities)

    def num_entities(self) -> int:
        return self.df_entities.shape[0]

    #############
    # RELATIONS #
    #############
    def _read_relations(self):
        # if no relations file exists
        if not Path(self.relations_filepath).is_file():
            print("Relations file does not exist.")
            df_relations = pd.DataFrame(columns=_RELATION_COLUMNS)
            df_relations["foodatlas_id"] = df_relations["foodatlas_id"].astype(int)

            return df_relations, 0

        # if relations file exists
        print(f"Loading relations file from {self.relations_filepath}...")
        df_relations = pd.read_csv(self.relations_filepath, sep="\t")
        df_relations["foodatlas_id"] = df_relations["foodatlas_id"].astype(int)
        df_relations.index = df_relations["foodatlas_id"].tolist()

        foodatlas_ids = df_relations["foodatlas_id"].tolist()
        avail_relation_id = 0 if len(foodatlas_ids) == 0 else max(foodatlas_ids) + 1

        return df_relations, avail_relation_id

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
            "foodatlas_id": self.avail_relation_id,
            "name": name,
            "translation": translation,
        }
        row = pd.Series(new_data, name=name)
        self.df_relations = pd.concat([self.df_relations, pd.DataFrame(row).transpose()])
        self.avail_relation_id += 1

        return row

    def get_all_relations(self) -> pd.DataFrame:
        return self.df_relations

    def get_relation_by_id(self, foodatlas_id: int) -> pd.Series:
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
    # assume you have an ontology.
    #        A                superclass
    #        |
    #     -------
    #     |     |
    #     B     C             class
    # (chemical B, isA, chemical A)
    # (chemical C, isA, chemical A)

    df = pd.DataFrame(
        [["chemical B", "isA", "chemical A"],
         ["chemical C", "isA", "chemical A"]],
        columns=["head", "relation", "tail"],
    )

    # and let's assume the metadata of nodes A, B, and C are as follows
    # A
    # type: chemical:superclass
    # name: chemical A
    # synonyms: [chem. A]
    # classyfire DB id: 123
    # Mesh ID: 567

    # B
    # type: chemical:class
    # name: chemical B
    # synonyms: [chem. B]
    # classyfire DB id: 723
    # Mesh ID: 427

    # C
    # type: chemical:class
    # name: chemical C
    # synonyms: []
    # classyfire DB id: 894
    # Mesh ID: 532

    # then we need to convert the head, relation, tail to
    # candidate types as follows.
    # you should probably do this in a for loop

    data = []

    # B isA A
    head = CandidateEntity(
        type="chemical:class",
        name="chemical B",
        synonyms=["chem. B"],
        other_db_ids={"ClassyFire": "723", "MESH": "427"},
    )

    relation = CandidateRelation(
        name="isA",
        translation="is a",
    )

    tail = CandidateEntity(
        type="chemical:superclass",
        name="chemical A",
        synonyms=["chem. A"],
        other_db_ids={"ClassyFire": "123", "MESH": "567"},
    )
    data.append([head, relation, tail])

    # C isA A
    head = CandidateEntity(
        type="chemical:class",
        name="chemical C",
        synonyms=[],
        other_db_ids={"ClassyFire": "894", "MESH": "532"},
    )

    relation = CandidateRelation(
        name="isA",
        translation="is a",
    )

    tail = CandidateEntity(
        type="chemical:superclass",
        name="chemical A",
        synonyms=["chem. A"],
        other_db_ids={"ClassyFire": "123", "MESH": "567"},
    )
    data.append([head, relation, tail])

    # you then populate the dataframe using this info
    df = pd.DataFrame(data, columns=["head", "relation", "tail"])
    print(df)

    # change this temporary directory to something you want !!!!!!!!
    temporary_dir = "/home/jasonyoun/Jason/Scratch/temp"
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(temporary_dir, "kg.txt"),
        evidence_filepath=os.path.join(temporary_dir, "evidence.txt"),
        entities_filepath=os.path.join(temporary_dir, "entities.txt"),
        relations_filepath=os.path.join(temporary_dir, "relations.txt"),
    )
    fa_kg.add_taxonomy(df, origin="ClassyFire")
    fa_kg.save()

    # check the files to understand the output format


if __name__ == '__main__':
    main()
