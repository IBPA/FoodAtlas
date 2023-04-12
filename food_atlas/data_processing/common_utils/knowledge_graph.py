from ast import literal_eval
from collections import namedtuple, OrderedDict, Counter
from copy import deepcopy, copy
import os
from pathlib import Path
import pickle
from typing import Dict, List, Any
import sys

from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
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
    "title",
    "section",
    "premise",
    "hypothesis",
    "source",
    "quality",
    "prob_mean",
    "prob_std",
]

_ENTITY_COLUMNS = [
    "foodatlas_id",
    "type",
    "name",
    "synonyms",
    "other_db_ids",
]

_RETIRED_ENTITY_COLUMNS = [
    "foodatlas_id",
    "type",
    "name",
    "synonyms",
    "other_db_ids",
    "reason",
]

_ENTITY_DEFAULTS = [
    None,
    None,
    None,
    [],
    {},
]

_FOOD_OTHER_DBS = [
    "NCBI_taxonomy",
    "foodatlas_part_id",
]
_CHEMICAL_OTHER_DBS = [
    "PubChem",
    "MESH",
    "MESH_tree",
    "CAS",
    "InChI",
    "InChIKey",
    "canonical_SMILES",
]
_ENTITY_OTHER_DBS = _FOOD_OTHER_DBS + _CHEMICAL_OTHER_DBS

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
            kg_dir: str,
            kg_filename: str = "kg.txt",
            evidence_filename: str = "evidence.txt",
            retired_entities_filename: str = "retired_entities.txt",
            entities_filename: str = "entities.txt",
            relations_filename: str = "relations.txt",
            nb_workers: int = None,
    ):
        # load the graph
        self.kg_filepath = os.path.join(kg_dir, kg_filename)
        self.df_kg = self._read_kg()

        # load evidence
        self.evidence_filepath = os.path.join(kg_dir, evidence_filename)
        self.df_evidence = self._read_evidence()

        # load the retired entities
        self.retired_entities_filepath = os.path.join(kg_dir, retired_entities_filename)
        self.df_retired_entities = self._read_retired_entities()

        # load the entities
        self.entities_filepath = os.path.join(kg_dir, entities_filename)
        self.df_entities, self.avail_entity_id = self._read_entities()

        # load the relations
        self.relations_filepath = os.path.join(kg_dir, relations_filename)
        self.df_relations, self.avail_relation_id = self._read_relations()

        if nb_workers is None:
            pandarallel.initialize(progress_bar=True)
        else:
            pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)

    def _read_kg(self) -> pd.DataFrame:
        if not Path(self.kg_filepath).is_file():
            print("KG file does not exist.")
            df_kg = pd.DataFrame(columns=_KG_COLUMNS)
            return df_kg

        print(f"Loading KG file from {self.kg_filepath}...")
        df_kg = pd.read_csv(self.kg_filepath, sep='\t', keep_default_na=False)
        assert set(df_kg.columns.tolist()) == set(_KG_COLUMNS)
        return df_kg[_KG_COLUMNS]

    def _read_evidence(self) -> pd.DataFrame:
        if not Path(self.evidence_filepath).is_file():
            print("Evidence file does not exist.")
            df_evidence = pd.DataFrame(columns=_EVIDENCE_COLUMNS)
            return df_evidence

        print(f"Loading evidence file from {self.evidence_filepath}...")
        df_evidence = pd.read_csv(
            self.evidence_filepath,
            sep='\t',
            keep_default_na=False,
            dtype={x: str for x in _EVIDENCE_COLUMNS}
        )
        assert set(df_evidence.columns.tolist()) == set(_EVIDENCE_COLUMNS)
        return df_evidence[_EVIDENCE_COLUMNS]

    def _read_retired_entities(self):
        if not Path(self.retired_entities_filepath).is_file():
            print("Retired entities file does not exist.")
            df_retired_entities = pd.DataFrame(columns=_RETIRED_ENTITY_COLUMNS)
            return df_retired_entities

        print(f"Loading retired entities file from {self.retired_entities_filepath}...")
        df_retired_entities = pd.read_csv(
            self.retired_entities_filepath,
            sep='\t',
            converters=_ENTITY_CONVERTERS,
        )
        return df_retired_entities[_RETIRED_ENTITY_COLUMNS]

    def _read_entities(self):
        if not Path(self.entities_filepath).is_file():
            print("Entities file does not exist.")
            df_entities = pd.DataFrame(columns=_ENTITY_COLUMNS)
            return df_entities, "e0"

        print(f"Loading entities file from {self.entities_filepath}...")
        df_entities = pd.read_csv(
            self.entities_filepath,
            sep="\t",
            converters=_ENTITY_CONVERTERS,
        )

        avail_entity_id = self._find_avail_foodatlas_id(
            df_entities["foodatlas_id"].tolist(), entity_or_relation="entity")

        # check integrity
        other_dbs = [y for x in df_entities["other_db_ids"].tolist() for y in x]
        assert set(other_dbs).issubset(_ENTITY_OTHER_DBS)
        return df_entities[_ENTITY_COLUMNS], avail_entity_id

    def _read_relations(self):
        if not Path(self.relations_filepath).is_file():
            print("Relations file does not exist.")
            df_relations = pd.DataFrame(columns=_RELATION_COLUMNS)
            return df_relations, "r0"

        print(f"Loading relations file from {self.relations_filepath}...")
        df_relations = pd.read_csv(self.relations_filepath, sep="\t")

        avail_relation_id = self._find_avail_foodatlas_id(
            df_relations["foodatlas_id"].tolist(), entity_or_relation="relation")

        return df_relations[_RELATION_COLUMNS], avail_relation_id

    def _find_avail_foodatlas_id(self, ids: List[str], entity_or_relation: str):
        prefix_str = list(set([x[0] for x in ids]))
        assert len(prefix_str) == 1
        prefix_str = prefix_str[0]

        if entity_or_relation == "entity":
            assert prefix_str == "e"
            ids = [int(x[1:]) for x in ids]

            # make sure that avail foodatlas id is not retired
            retired_ids = [int(x[1:]) for x in self.df_retired_entities["foodatlas_id"].tolist()]
            avail_id = max(ids) + 1
            if len(retired_ids) != 0:
                while avail_id <= max(retired_ids):
                    avail_id += 1

            return f"{prefix_str}{avail_id}"
        elif entity_or_relation == "relation":
            assert prefix_str == "r"
            ids = [int(x[1:]) for x in ids]
            return f"{prefix_str}{max(ids) + 1}"
        else:
            raise ValueError()

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

    def _check_duplicate_evidence(self, evidence):
        raise NotImplementedError()

    def add_triples(self, df: pd.DataFrame, source=None):
        assert set(["head", "relation", "tail", "source", "quality"]).issubset(df.columns)
        assert source in [None, 'lp']
        print(f"Size of triples to add: {df.shape[0]}")

        if source is None:
            entities = df["head"].tolist() + df["tail"].tolist()

            # check all chemical entities integrity
            chemical_entities = [e for e in entities if e.type == "chemical"]
            for e in chemical_entities:
                assert "PubChem" in e.other_db_ids or "MESH" in e.other_db_ids
                if "PubChem" in e.other_db_ids:
                    assert len(e.other_db_ids["PubChem"]) == 1
                if "MESH" in e.other_db_ids:
                    assert len(e.other_db_ids["MESH"]) != 0
            self.add_update_entities(entities)

            relations = df["relation"].tolist()
            self._add_candidate_relations(relations)

        print("Adding triples...")

        def _f(row):
            data = []
            evidence = []

            if source == 'lp':
                head_foodatlas_id = row["head"]
                relation_foodatlas_id = row["relation"]
                tail_foodatlas_id = row["tail"]
            else:
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
                "pmid": row["pmid"] if "pmid" in row.index else '',
                "pmcid": row["pmcid"] if "pmcid" in row.index else '',
                "title": row["title"] if "title" in row.index else '',
                "section": row["section"] if "section" in row.index else '',
                "premise": row["premise"] if "premise" in row.index else '',
                "hypothesis":
                    row["hypothesis_string"] if "hypothesis_string" in row.index else '',
                "source": row["source"],
                "quality": row["quality"],
                "prob_mean": row["prob_mean"] if "prob_mean" in row.index else '',
                "prob_std": row["prob_std"] if "prob_std" in row.index else '',
            })
            evidence.append(e)

            return data, evidence

        data, evidence = [], []
        for d, e in df.parallel_apply(_f, axis=1):
            data.extend(d)
            evidence.extend(e)

        print()
        print("Updating the KG...")
        self.df_kg = pd.concat([self.df_kg, pd.DataFrame(data)])
        print("Dropping duplicates in the KG...")
        self.df_kg.drop_duplicates(inplace=True, ignore_index=True)

        print("Updating the evidence...")
        self.df_evidence = pd.concat([self.df_evidence, pd.DataFrame(evidence)])
        self.df_evidence = self.df_evidence.astype('str')
        print("Dropping duplicates in the evidence...")
        self.df_evidence.drop_duplicates(inplace=True, ignore_index=True)

        return self.df_kg, self.df_evidence

    def add_update_entities(
            self,
            entities: List[CandidateEntity],
    ) -> pd.Series:
        print(f"Number of entities before updating: {self.df_entities.shape[0]}")

        print("Preparing to add entities...")
        entities = [eval(e, globals()) for e in sorted(set([str(e) for e in entities]))]
        data = [["", e.type, e.name, e.synonyms + [e.name], e.other_db_ids] for e in entities]
        df_entities_to_add = pd.DataFrame(data, columns=_ENTITY_COLUMNS)
        df_entities = pd.concat([self._nested_deep_copy(self.df_entities), df_entities_to_add])
        df_entities.reset_index(inplace=True, drop=True)

        duplicate_idx = []

        # organisms and organism with part
        print("Finding duplicate organisms...")
        df_organisms = self.get_entities_by_type(
            df=df_entities,
            startswith_type="organism",
        )

        def _organism_unique_id(other_db_ids):
            assert "NCBI_taxonomy" in other_db_ids and \
                len(other_db_ids["NCBI_taxonomy"]) == 1 and \
                "foodatlas_part_id" in other_db_ids
            return other_db_ids["NCBI_taxonomy"][0] + '-' + other_db_ids["foodatlas_part_id"]

        if df_organisms.shape[0] > 0:
            df_organisms = self._nested_deep_copy(df_organisms)
            df_organisms["_unique_id"] = df_organisms["other_db_ids"].apply(
                lambda x: _organism_unique_id(x))
            for _, df_subset in df_organisms.groupby("_unique_id"):
                if df_subset.shape[0] > 1:
                    duplicate_idx.append(list(df_subset.index))

        # chemicals
        print("Finding duplicate chemicals...")
        df_chemicals = self.get_entities_by_type(
            df=df_entities,
            exact_type="chemical",
        )

        # chemicals with PubChem
        df_chemicals_with_pubchem = df_chemicals[df_chemicals["other_db_ids"].apply(
            lambda x: "PubChem" in x)]

        if df_chemicals_with_pubchem.shape[0] > 0:
            df_chemicals_with_pubchem = self._nested_deep_copy(df_chemicals_with_pubchem)
            df_chemicals_with_pubchem["PubChem"] = \
                df_chemicals_with_pubchem["other_db_ids"].apply(lambda x: x["PubChem"])
            df_chemicals_with_pubchem = df_chemicals_with_pubchem.explode("PubChem")
            for _, df_subset in df_chemicals_with_pubchem.groupby("PubChem"):
                if df_subset.shape[0] > 1:
                    duplicate_idx.append(list(df_subset.index))

        # chemicals without PubChem (for MeSH ontology or entities without matching PubChem ID)
        df_chemicals_without_pubchem = df_chemicals[df_chemicals["other_db_ids"].apply(
            lambda x: "PubChem" not in x)]

        if df_chemicals_without_pubchem.shape[0] > 0:
            df_chemicals_without_pubchem = self._nested_deep_copy(df_chemicals_without_pubchem)
            df_chemicals_without_pubchem["MESH"] = \
                df_chemicals_without_pubchem["other_db_ids"].apply(lambda x: x["MESH"])
            df_chemicals_without_pubchem = df_chemicals_without_pubchem.explode("MESH")
            for _, df_subset in df_chemicals_without_pubchem.groupby("MESH"):
                if df_subset.shape[0] > 1:
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
        entities_to_merge = {"merge_from": [], "merge_to": []}
        for idx in duplicate_idx:
            df_duplicates = df_entities.iloc[idx]
            _type = df_duplicates.iloc[0]["type"]

            ids = list(set(df_duplicates["foodatlas_id"].tolist()))
            ids = [x for x in ids if x != ""]

            if len(ids) == 0:
                foodatlas_id = ""
                name = df_duplicates.iloc[0]["name"]
            elif len(ids) == 1:
                foodatlas_id = ids[0]
                name = df_duplicates[df_duplicates["foodatlas_id"] != ""].iloc[0]["name"]
            else:
                ids = sorted(ids, key=lambda x: int("".join([i for i in x if i.isdigit()])))
                foodatlas_id = ids[0]
                name = df_duplicates[df_duplicates["foodatlas_id"] != ""].iloc[0]["name"]

                for x in ids[1:]:
                    entities_to_merge["merge_from"].append(x)
                    entities_to_merge["merge_to"].append(foodatlas_id)

            assert name is not None
            names = df_duplicates["name"].tolist()
            names = [x for x in names if x is not None]

            synonyms = [x for subset in df_duplicates["synonyms"].tolist() for x in subset]
            to_merge = [x for x in list(set(names + synonyms)) if x is not None]
            synonyms = KnowledgeGraph._merge_synonyms(to_merge)

            other_db_ids = {}
            foodatlas_part_ids = []
            for d in df_duplicates["other_db_ids"].tolist():
                for k, v in d.items():
                    if k != "foodatlas_part_id":
                        assert type(v) == list
                        if k not in other_db_ids:
                            other_db_ids[k] = v
                        else:
                            other_db_ids[k].extend(v)
                    else:
                        assert type(v) == str
                        foodatlas_part_ids.append(v)
            other_db_ids = {k: sorted(set(v)) for k, v in other_db_ids.items()}
            if len(foodatlas_part_ids) != 0:
                foodatlas_part_ids = list(set(foodatlas_part_ids))
                assert len(foodatlas_part_ids) == 1
                other_db_ids["foodatlas_part_id"] = foodatlas_part_ids[0]

            merged_duplicates.append([foodatlas_id, _type, name, synonyms, other_db_ids])

        flattened_idx = list(set([y for x in duplicate_idx for y in x]))
        non_duplicate_idx = [x for x in df_entities.index if x not in flattened_idx]

        # update retired entities
        df_entities_to_merge = pd.DataFrame(entities_to_merge)
        df_entities_to_merge.drop_duplicates(inplace=True)
        if df_entities_to_merge.shape[0] > 0:
            self._update_retired_entities(df_entities_to_merge, reason="merge")

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

    def _update_retired_entities(self, _input, reason):
        if reason == "merge":
            assert type(_input) == pd.DataFrame
            entities_to_remove = list(set(_input["merge_from"].tolist()))
            _input_dict = dict(zip(_input["merge_from"].tolist(), _input["merge_to"].tolist()))

            df_retired_entities = self.df_entities[self.df_entities["foodatlas_id"].apply(
                lambda x: x in entities_to_remove
            )]
            df_retired_entities["reason"] = df_retired_entities["foodatlas_id"].apply(
                lambda x: f"Merged to {_input_dict[x]}"
            )
            df_retired_entities = df_retired_entities[_RETIRED_ENTITY_COLUMNS]
            self.df_retired_entities = pd.concat([self.df_retired_entities, df_retired_entities])

            # kg
            self.df_kg["head"] = self.df_kg["head"].replace(_input_dict)
            self.df_kg["tail"] = self.df_kg["tail"].replace(_input_dict)

            # evidence
            self.df_evidence["head"] = self.df_evidence["head"].replace(_input_dict)
            self.df_evidence["tail"] = self.df_evidence["tail"].replace(_input_dict)
            self.df_evidence["triple"] = self.df_evidence.apply(
                lambda row: f"({row['head']},{row['relation']},{row['tail']})", axis=1)
        else:
            raise NotImplementedError

    def _add_candidate_relations(
            self,
            relations: List[CandidateRelation],
    ) -> pd.Series:
        print(f"Number of relations before updating: {self.df_relations.shape[0]}")

        relations = [eval(r, globals()) for r in sorted(set([str(r) for r in relations]))]
        data = [["", r.name, r.translation] for r in relations]
        df_relations = pd.concat([
            self._nested_deep_copy(self.df_relations),
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
            if entity_or_relation.type.startswith("organism"):
                df_match = self.get_entity_by_other_db_id(entity_or_relation.other_db_ids)
            elif entity_or_relation.type == "chemical":
                if "PubChem" in entity_or_relation.other_db_ids:
                    other_db_ids = {"PubChem": entity_or_relation.other_db_ids["PubChem"]}
                elif "MESH" in entity_or_relation.other_db_ids:
                    other_db_ids = {"MESH": entity_or_relation.other_db_ids["MESH"]}
                else:
                    raise ValueError()
                df_match = self.get_entity_by_other_db_id(other_db_ids)
        elif type(entity_or_relation) == CandidateRelation:
            df_match = self.df_relations[self.df_relations["name"] == entity_or_relation.name]
        else:
            raise ValueError()

        assert df_match.shape[0] == 1
        return df_match.iloc[0]["foodatlas_id"]

    @staticmethod
    def _merge_synonyms(synonyms):
        all_synonyms = set(synonyms)
        seen = OrderedDict()
        for word in all_synonyms:
            lo = word.lower()
            seen[lo] = min(word, seen.get(lo, word))
        return list(seen.values())

    def _overwrite_entity(
            self,
            foodatlas_id: str,
            type_: str,
            name: str = None,
            synonyms: List[str] = [],
            other_db_ids: Dict[str, Any] = {},
    ):
        self.df_entities.set_index("foodatlas_id", drop=False, inplace=True)

        self.df_entities.at[foodatlas_id, "type"] = type_
        if type_.startswith("organism_with_part"):
            part_name = self.df_entities.at[foodatlas_id, "name"].split(" - ")[-1]

        if name:
            all_names = self._nested_deep_copy(self.df_entities.at[foodatlas_id, "synonyms"])
            all_names += [self.df_entities.at[foodatlas_id, "name"]]
            all_names += [name]
            all_names = list(set(all_names))

            self.df_entities.at[foodatlas_id, "name"] = name
            self.df_entities.at[foodatlas_id, "synonyms"] = \
                KnowledgeGraph._merge_synonyms(all_names)

        if synonyms:
            all_names = self._nested_deep_copy(self.df_entities.at[foodatlas_id, "synonyms"])
            if type_.startswith("organism_with_part"):
                all_names += [f"{x} - {part_name}" for x in synonyms]
            else:
                all_names += synonyms
            all_names = set(all_names)

            self.df_entities.at[foodatlas_id, "synonyms"] = \
                KnowledgeGraph._merge_synonyms(all_names)

        if other_db_ids:
            self.df_entities.at[foodatlas_id, "other_db_ids"] = other_db_ids

        return self.df_entities.loc[foodatlas_id]

    def get_kg(self) -> pd.DataFrame:
        return self._nested_deep_copy(self.df_kg)

    def get_kg_using_relation_name(self, relation_name: str) -> pd.DataFrame:
        relation = self.get_relation_by_name(relation_name)
        relation_foodatlas_id = relation["foodatlas_id"]
        df_kg = self.df_kg[self.df_kg["relation"] == relation_foodatlas_id]
        return self._nested_deep_copy(df_kg)

    def get_evidence(self) -> pd.DataFrame:
        return self._nested_deep_copy(self.df_evidence)

    def get_all_entities(self) -> pd.DataFrame:
        return self._nested_deep_copy(self.df_entities)

    def get_entity_by_id(self, foodatlas_id: str) -> pd.Series:
        entity = self.df_entities[self.df_entities["foodatlas_id"] == foodatlas_id]
        assert entity.shape[0] == 1
        return self._nested_deep_copy(entity.iloc[0])

    def get_entity_by_other_db_id(self, other_db_ids) -> pd.DataFrame:
        entity = self.df_entities[self.df_entities["other_db_ids"].apply(
            lambda x: other_db_ids.items() <= x.items())]
        return self._nested_deep_copy(entity)

    def _nested_deep_copy(self, x):
        return pickle.loads(pickle.dumps(x))

    def get_entity_by_name(self, name: str) -> pd.Series:
        raise NotImplementedError()

    def get_entities_by_type(
            self,
            df: pd.DataFrame = None,
            exact_type: str = None,
            startswith_type: str = None,
    ) -> pd.DataFrame:
        if df is not None:
            df_to_filter = self._nested_deep_copy(df)
        else:
            df_to_filter = self._nested_deep_copy(self.df_entities)

        data = []
        if exact_type:
            data.append(
                df_to_filter[df_to_filter["type"].apply(lambda x: x == exact_type)])
        if startswith_type:
            data.append(
                df_to_filter[df_to_filter["type"].apply(lambda x: x.startswith(startswith_type))])

        if len(data) == 0:
            return None
        else:
            return pd.concat(data)

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
        return self._nested_deep_copy(self.df_relations)

    def get_relation_by_id(self, foodatlas_id: str) -> pd.Series:
        relation = self.df_relations[self.df_relations["foodatlas_id"] == foodatlas_id]
        assert relation.shape[0] == 1
        return self._nested_deep_copy(relation.iloc[0])

    def get_relation_by_name(self, name: str) -> pd.Series:
        relation = self.df_relations[self.df_relations["name"] == name]
        assert relation.shape[0] == 1
        return self._nested_deep_copy(relation.iloc[0])

    def print_all_relations(self) -> None:
        print(self.df_relations)

    def num_relations(self) -> int:
        return self.df_relations.shape[0]

    def save(
            self,
            kg_dir: str = None,
            kg_filename: str = "kg.txt",
            evidence_filename: str = "evidence.txt",
            retired_entities_filename: str = "retired_entities.txt",
            entities_filename: str = "entities.txt",
            relations_filename: str = "relations.txt",
    ) -> None:
        self.df_entities["foodatlas_id_val_only"] = \
            self.df_entities["foodatlas_id"].apply(lambda x: int(x[1:]))
        self.df_entities.sort_values(
            "foodatlas_id_val_only", inplace=True, ignore_index=True)

        self.df_retired_entities["foodatlas_id_val_only"] = \
            self.df_retired_entities["foodatlas_id"].apply(lambda x: int(x[1:]))
        self.df_retired_entities.sort_values(
            "foodatlas_id_val_only", inplace=True, ignore_index=True)

        self.df_relations["foodatlas_id_val_only"] = \
            self.df_relations["foodatlas_id"].apply(lambda x: int(x[1:]))
        self.df_relations.sort_values(
            "foodatlas_id_val_only", inplace=True, ignore_index=True)

        self.df_entities.drop("foodatlas_id_val_only", inplace=True, axis=1)
        self.df_retired_entities.drop("foodatlas_id_val_only", inplace=True, axis=1)
        self.df_relations.drop("foodatlas_id_val_only", inplace=True, axis=1)

        if kg_dir is not None:
            Path(kg_dir).mkdir(parents=True, exist_ok=True)

        if kg_dir is not None:
            kg_filepath = os.path.join(kg_dir, kg_filename)
            print(f"Saving kg to a new filepath: {kg_filepath}")
            self.df_kg.to_csv(kg_filepath, sep='\t', index=False)
        else:
            print(f"Saving kg to original filepath: {self.kg_filepath}")
            self.df_kg.to_csv(self.kg_filepath, sep='\t', index=False)

        if kg_dir is not None:
            evidence_filepath = os.path.join(kg_dir, evidence_filename)
            print(f"Saving evidence to a new filepath: {evidence_filepath}")
            self.df_evidence.to_csv(evidence_filepath, sep='\t', index=False)
        else:
            print(f"Saving evidence to original filepath: {self.evidence_filepath}")
            self.df_evidence.to_csv(self.evidence_filepath, sep='\t', index=False)

        if kg_dir is not None:
            retired_entities_filepath = os.path.join(kg_dir, retired_entities_filename)
            print(f"Saving retired entities to a new filepath: {retired_entities_filepath}")
            self.df_retired_entities.to_csv(retired_entities_filepath, sep='\t', index=False)
        else:
            print(f"Saving retired entities to original filepath: {self.retired_entities_filepath}")
            self.df_retired_entities.to_csv(self.retired_entities_filepath, sep='\t', index=False)

        if kg_dir is not None:
            entities_filepath = os.path.join(kg_dir, entities_filename)
            print(f"Saving entities to a new filepath: {entities_filepath}")
            self.df_entities.to_csv(entities_filepath, sep='\t', index=False)
        else:
            print(f"Saving entities to original filepath: {self.entities_filepath}")
            self.df_entities.to_csv(self.entities_filepath, sep='\t', index=False)

        if kg_dir is not None:
            relations_filepath = os.path.join(kg_dir, relations_filename)
            print(f"Saving relations to a new filepath: {relations_filepath}")
            self.df_relations.to_csv(relations_filepath, sep='\t', index=False)
        else:
            print(f"Saving relations to original filepath: {self.relations_filepath}")
            self.df_relations.to_csv(self.relations_filepath, sep='\t', index=False)


def main():
    raise NotImplementedError()


if __name__ == '__main__':
    main()
