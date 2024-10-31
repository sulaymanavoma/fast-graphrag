# type: ignore
import unittest
from unittest.mock import AsyncMock, MagicMock, call

from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._graph_upsert import (
    DefaultEdgeUpsertPolicy,
    # DefaultGraphUpsertPolicy,
    DefaultNodeUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._storage._base import BaseGraphStorage


class TestNodeUpsertPolicy_SummarizeDescription(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call_same_node_summarize(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "This is a lengthy random description."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "This is a lengthy random description that is being used to test the summarization."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node = AsyncMock(return_value=(None, None))
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_size=len(node1.description) + 4,
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)
        self.assertEqual(upserted_nodes[0][1].name, "node1")
        self.assertEqual(upserted_nodes[0][1].description, "This is a summary.")

    async def test_call_same_node_no_summarize(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "This is a short random description 1."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "This is a short random description 2."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node.side_effect = [(None, None), (node1, 0)]
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_size=(len(node1.description) * 2) + 4
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)

        self.assertEqual(
            upserted_nodes[0][1].description,
            "This is a short random description 1.\nThis is a short random description 2.",
        )

        # Assertions
        llm.send_message.assert_not_called()

    async def test_call_two_nodes(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node2"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node.side_effect = [(None, None), (None, None)]
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_size=len(node1.description) + 4,
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)

        self.assertEqual(upserted_nodes[0][1].description, "Description for node1.")
        self.assertEqual(upserted_nodes[1][1].description, "Description for node2.")


class TestEdgeUpsertPolicy_UpsertIfValidNodes(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        edge1 = MagicMock()
        edge1.source = "source1"
        edge1.target = "target1"
        edge2 = MagicMock()
        edge2.source = "source1"
        edge2.target = "target2"
        source_edges = [edge1, edge2]

        # Mock methods
        target.get_node.side_effect = lambda x: (MagicMock(), None) if x in ["source1", "target1"] else (None, None)
        target.upsert_edge = AsyncMock()

        # Create policy instance
        policy = EdgeUpsertPolicy_UpsertIfValidNodes(config=EdgeUpsertPolicy_UpsertIfValidNodes.Config())

        # Call the method
        await policy(llm, target, source_edges)

        # Assertions
        target.get_node.assert_has_calls([call("source1"), call("target1"), call("source1"), call("target2")])
        target.upsert_edge.assert_called_once_with(edge=edge1, edge_index=None)


class TestDefaultNodeUpsertPolicy(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call_same_id(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        target.get_node = AsyncMock()
        target.get_node.side_effect = [(None, None), (node1, 0)]
        target.upsert_node = AsyncMock()
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = DefaultNodeUpsertPolicy(config=None)

        # Call the method
        _, upserted = await policy(llm, target, source_nodes)
        upserted = list(upserted)

        self.assertEqual(len(upserted), 1)
        self.assertEqual(upserted[0][1].description, "Description for node2.")

        # Assertions
        target.get_node.assert_has_calls([call(node1), call(node2)])
        target.upsert_node.assert_has_calls([call(node=node1, node_index=None), call(node=node2, node_index=0)])

    async def test_call_different_id(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node2"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        target.get_node = AsyncMock()
        target.get_node.side_effect = [(None, None), (None, None)]
        target.upsert_node = AsyncMock()
        target.upsert_node.side_effect = [0, 1]  # type: ignore

        # Create policy instance
        policy = DefaultNodeUpsertPolicy(config=None)

        # Call the method
        _, upserted = await policy(llm, target, source_nodes)
        upserted = list(upserted)
        self.assertEqual(len(upserted), 2)

        # Assertions
        target.get_node.assert_has_calls([call(node1), call(node2)])
        target.upsert_node.assert_has_calls([call(node=node1, node_index=None), call(node=node2, node_index=None)])


class TestDefaultEdgeUpsertPolicy(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        edge = MagicMock()
        source_edges = [edge]

        # Mock methods
        target.upsert_edge = AsyncMock()

        # Create policy instance
        policy = DefaultEdgeUpsertPolicy(config=None)

        # Call the method
        await policy(llm, target, source_edges)

        # Assertions
        target.upsert_edge.assert_called_once_with(edge=edge, edge_index=None)


# class TestDefaultGraphUpsertPolicy(unittest.IsolatedAsyncioTestCase):
#     async def test_call(self):
#         # Mock dependencies
#         llm = AsyncMock(spec=BaseLLMService)
#         source = AsyncMock(spec=BaseGraphStorage)
#         target_nodes = AsyncMock(spec=BaseGraphStorage)
#         target_edges = AsyncMock(spec=BaseGraphStorage)
#         node = MagicMock()
#         edge = MagicMock()
#         source_nodes = [node]
#         source_edges = [edge]

#         # Mock methods
#         policy = DefaultGraphUpsertPolicy[TEntity, TRelation, TId](
#             config=None,
#             nodes_upsert_cls=DefaultNodeUpsertPolicy[TEntity, TId],
#             edges_upsert_cls=DefaultEdgeUpsertPolicy[TRelation, TId],
#         )
#         policy._nodes_upsert = AsyncMock(return_value=target_nodes)
#         policy._edges_upsert = AsyncMock(return_value=target_edges)

#         # Call the method
#         result = await policy(llm, source, source_nodes, source_edges)

#         # Assertions
#         policy._nodes_upsert.assert_called_once_with(llm, source, source_nodes)
#         policy._edges_upsert.assert_called_once_with(llm, target_nodes, source_edges)
#         self.assertEqual(result, target_edges)

#     async def test_call_with_default(self):
#         pass


if __name__ == "__main__":
    unittest.main()
