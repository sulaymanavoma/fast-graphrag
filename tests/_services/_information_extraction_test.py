# type: ignore
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fast_graphrag._llm._base import BaseLLMService
from fast_graphrag._policies._graph_upsert import BaseGraphUpsertPolicy
from fast_graphrag._services import DefaultInformationExtractionService
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import TGraph, TQueryEntities


class TestDefaultInformationExtractionService(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm_service = MagicMock(spec=BaseLLMService)
        self.llm_service.send_message = AsyncMock()
        self.chunk = MagicMock()
        self.chunk.content = "test content"
        self.chunk.id = "chunk_id"
        self.document = [self.chunk]
        self.prompt_kwargs = {"domain": "test_domain"}
        self.service = DefaultInformationExtractionService(
            graph_upsert=None
        )
        self.service.graph_upsert = AsyncMock(spec=BaseGraphUpsertPolicy)

    @patch('fast_graphrag._services._information_extraction.format_and_send_prompt', new_callable=AsyncMock)
    async def test_extract_entities_from_query(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (TQueryEntities(entities=["entity1", "entity2"], n=2), None)
        entities = await self.service.extract_entities_from_query(self.llm_service, "test query", self.prompt_kwargs)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0].name, "ENTITY1")
        self.assertEqual(entities[1].name, "ENTITY2")

    @patch('fast_graphrag._services._information_extraction.format_and_send_prompt', new_callable=AsyncMock)
    async def test_extract_from_chunk(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (TGraph(entities=[], relationships=[]), [])
        graph = await self.service._extract_from_chunk(self.llm_service, self.chunk, self.prompt_kwargs)
        self.assertIsInstance(graph, TGraph)

    @patch('fast_graphrag._services._information_extraction.format_and_send_prompt', new_callable=AsyncMock)
    async def test_extract(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (TGraph(entities=[], relationships=[]), [])
        tasks = self.service.extract(self.llm_service, [self.document], self.prompt_kwargs)
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BaseGraphStorage)

    @patch('fast_graphrag._services._information_extraction.format_and_send_prompt', new_callable=AsyncMock)
    async def test_merge(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (TGraph(entities=[], relationships=[]), [])
        graph_storage = await self.service._merge(self.llm_service, [TGraph(entities=[], relationships=[])])
        self.assertIsInstance(graph_storage, BaseGraphStorage)

if __name__ == '__main__':
    unittest.main()
