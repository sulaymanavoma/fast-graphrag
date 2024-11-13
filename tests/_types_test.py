# type: ignore
import unittest
from dataclasses import asdict

from pydantic import ValidationError

from fast_graphrag._types import (
    TChunk,
    TContext,
    TDocument,
    TEditRelation,
    TEditRelationList,
    TEntity,
    TGraph,
    TQueryEntities,
    TQueryResponse,
    TRelation,
    TScore,
    dump_to_csv,
)


class TestTypes(unittest.TestCase):
    def test_tdocument(self):
        doc = TDocument(data="Sample data", metadata={"key": "value"})
        self.assertEqual(doc.data, "Sample data")
        self.assertEqual(doc.metadata, {"key": "value"})

    def test_tchunk(self):
        chunk = TChunk(id=123, content="Sample content", metadata={"key": "value"})
        self.assertEqual(chunk.id, 123)
        self.assertEqual(chunk.content, "Sample content")
        self.assertEqual(chunk.metadata, {"key": "value"})

    def test_tentity(self):
        entity = TEntity(name="Entity1", type="Type1", description="Description1")
        self.assertEqual(entity.name, "Entity1")
        self.assertEqual(entity.type, "Type1")
        self.assertEqual(entity.description, "Description1")

        pydantic_entity = TEntity.Model(name="Entity1", type="Type1", description="Description1")
        entity.name = entity.name.upper()
        entity.type = entity.type.upper()
        self.assertEqual(asdict(entity), asdict(pydantic_entity.to_dataclass(pydantic_entity)))

    def test_tqueryentities(self):
        query_entities = TQueryEntities(entities=["Entity1", "Entity2"], n=2)
        self.assertEqual(query_entities.entities, ["ENTITY1", "ENTITY2"])
        self.assertEqual(query_entities.n, 2)

        with self.assertRaises(ValidationError):
            TQueryEntities(entities=["Entity1", "Entity2"], n="two")

    def test_trelation(self):
        relation = TRelation(source="Entity1", target="Entity2", description="Relation description")
        self.assertEqual(relation.source, "Entity1")
        self.assertEqual(relation.target, "Entity2")
        self.assertEqual(relation.description, "Relation description")

        pydantic_relation = TRelation.Model(
            source_entity="Entity1", target_entity="Entity2", description="Relation description"
        )

        relation.source = relation.source.upper()
        relation.target = relation.target.upper()
        self.assertEqual(asdict(relation), asdict(pydantic_relation.to_dataclass(pydantic_relation)))

    def test_tgraph(self):
        entity = TEntity(name="Entity1", type="Type1", description="Description1")
        relation = TRelation(source="Entity1", target="Entity2", description="Relation description")
        graph = TGraph(entities=[entity], relationships=[relation])
        self.assertEqual(graph.entities, [entity])
        self.assertEqual(graph.relationships, [relation])

        pydantic_graph = TGraph.Model(
            entities=[TEntity.Model(name="Entity1", type="Type1", description="Description1")],
            relationships=[
                TRelation.Model(source_entity="Entity1", target_entity="Entity2", description="Relation description")
            ],
        )

        for entity in graph.entities:
            entity.name = entity.name.upper()
            entity.type = entity.type.upper()
        for relation in graph.relationships:
            relation.source = relation.source.upper()
            relation.target = relation.target.upper()
        self.assertEqual(asdict(graph), asdict(pydantic_graph.to_dataclass(pydantic_graph)))

    def test_teditrelationship(self):
        edit_relationship = TEditRelation(ids=[1, 2], description="Combined relationship description")
        self.assertEqual(edit_relationship.ids, [1, 2])
        self.assertEqual(edit_relationship.description, "Combined relationship description")

    def test_teditrelationshiplist(self):
        edit_relationship = TEditRelation(ids=[1, 2], description="Combined relationship description")
        edit_relationship_list = TEditRelationList(grouped_facts=[edit_relationship])
        self.assertEqual(edit_relationship_list.groups, [edit_relationship])

    def test_tcontext(self):
        context = TContext(
            entities=[("Entity1", TScore(0.9))],
            relationships=[("Relation1", TScore(0.8))],
            chunks=[("Chunk1", TScore(0.7))],
        )
        self.assertEqual(context.entities, [("Entity1", TScore(0.9))])
        self.assertEqual(context.relationships, [("Relation1", TScore(0.8))])
        self.assertEqual(context.chunks, [("Chunk1", TScore(0.7))])

    def test_tqueryresponse(self):
        context = TContext(
            entities=[("Entity1", TScore(0.9))],
            relationships=[("Relation1", TScore(0.8))],
            chunks=[("Chunk1", TScore(0.7))],
        )
        query_response = TQueryResponse(response="Sample response", context=context)
        self.assertEqual(query_response.response, "Sample response")
        self.assertEqual(query_response.context, context)

    def test_dump_to_csv(self):
        data = [TEntity(name="Sample name", type="SAMPLE TYPE", description="Sample description")]
        fields = ["name", "type"]
        values = {"score": [0.9]}
        csv_output = dump_to_csv(data, fields, with_header=True, **values)
        expected_output = "```csv\nname;;type;;score\nSample name;;SAMPLE TYPE;;0.9\n```"
        self.assertEqual(csv_output, expected_output)


if __name__ == "__main__":
    unittest.main()
