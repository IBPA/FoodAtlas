import os
import pandas as pd
from pandarallel import pandarallel


class FoodAtlasKnowledgeGraph:
    """
    """
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

    def add_triplets(self):
        pass

    def add_update_entities(self):
        pass
