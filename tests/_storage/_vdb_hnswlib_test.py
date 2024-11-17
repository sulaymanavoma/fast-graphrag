# type: ignore
import pickle
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig, InvalidStorageError


class TestHNSWVectorStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = HNSWVectorStorageConfig()
        self.storage = HNSWVectorStorage(config=self.config, embedding_dim=128)
        self.storage._index = MagicMock()
        self.storage._index.get_current_count.return_value = 0

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_upsert(self, mock_logger):
        ids = [1, 2, 3]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        metadata = [{"meta1": "data1"}, {"meta2": "data2"}, {"meta3": "data3"}]

        self.storage._current_elements = 0
        self.storage._index.get_current_count.return_value = 3

        await self.storage.upsert(ids, embeddings, metadata)

        self.assertEqual(self.storage._current_elements, 3)
        self.assertEqual(self.storage._metadata[1], {"meta1": "data1"})
        self.assertEqual(self.storage._metadata[2], {"meta2": "data2"})
        self.assertEqual(self.storage._metadata[3], {"meta3": "data3"})

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_upsert_full_index(self, mock_logger):
        ids = [1, 2, 3]
        embeddings = np.random.rand(3, 128).astype(np.float32)

        self.storage._current_elements = self.config.max_elements

        with self.assertRaises(NotImplementedError):
            await self.storage.upsert(ids, embeddings)

        mock_logger.error.assert_called_with("HNSW index is full. Cannot insert 3 elements.")

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_get_knn_empty_index(self, mock_logger):
        embeddings = np.random.rand(1, 128).astype(np.float32)

        ids, distances = await self.storage.get_knn(embeddings, top_k=5)

        self.assertEqual(ids, [])
        self.assertTrue(np.array_equal(distances, np.array([], dtype=np.float32)))
        mock_logger.info.assert_called_with("Querying knns in empty index.")

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_get_knn(self, mock_logger):
        embeddings = np.random.rand(2, 128).astype(np.float32)
        self.storage._current_elements = 10
        self.storage._index.knn_query.return_value = ([[1, 2, 3], [4, 5, 6]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        ids, distances = await self.storage.get_knn(embeddings, top_k=3)

        self.storage._index.knn_query.assert_called_once_with(data=embeddings, k=3, num_threads=self.config.num_threads)
        self.assertEqual(ids, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_almost_equal(distances, np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32))

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_score_all_empty_index(self, mock_logger):
        embeddings = np.random.rand(1, 128).astype(np.float32)

        scores = await self.storage.score_all(embeddings, top_k=1)

        self.assertEqual(scores.shape, (0, 0))
        mock_logger.warning.assert_called_with("No provided embeddings (128) or empty index (0).")

    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_score_all(self, mock_logger):
        embeddings = np.random.rand(2, 128).astype(np.float32)
        self.storage._current_elements = 10
        self.storage._index.knn_query.return_value = ([[1, 2, 3]], [[0.1, 0.2, 0.3]])

        scores = await self.storage.score_all(embeddings, top_k=3)

        self.storage._index.knn_query.assert_called_once_with(data=embeddings, k=3, num_threads=self.config.num_threads)
        self.assertEqual(scores.shape, (1, 10))
        np.testing.assert_almost_equal(scores.data, np.array([0.95, 0.9, 0.85], dtype=np.float32))

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps(({"key": "value"}, 3)))
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_insert_start_with_existing_files(self, mock_logger, mock_open, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.side_effect = lambda x: f"dummy_path_{x}"

        await self.storage._insert_start()

        mock_index.assert_called_with(space="cosine", dim=self.storage.embedding_dim)
        self.storage._index.load_index.assert_called_with(
            "dummy_path_hnsw_index_128.bin", max_elements=self.config.max_elements
        )
        self.assertEqual(self.storage._metadata, {"key": "value"})
        self.assertEqual(self.storage._current_elements, 3)
        mock_logger.debug.assert_called_with("Loaded 3 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_insert_start_with_no_files(self, mock_logger, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None

        await self.storage._insert_start()

        mock_index.assert_called_with(space="cosine", dim=self.storage.embedding_dim)
        self.storage._index.init_index.assert_called_with(
            max_elements=self.config.max_elements,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
        )
        self.storage._index.set_ef.assert_called_with(self.config.ef_search)
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage._current_elements, 0)

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_insert_start_with_no_namespace(self, mock_logger, mock_index):
        self.storage.namespace = None

        await self.storage._insert_start()

        mock_index.assert_called_with(space="cosine", dim=self.storage.embedding_dim)
        self.storage._index.init_index.assert_called_with(
            max_elements=self.config.max_elements,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
        )
        self.storage._index.set_ef.assert_called_with(self.config.ef_search)
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage._current_elements, 0)
        mock_logger.debug.assert_called_with("Creating new volatile vectordb storage.")

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_insert_done(self, mock_logger, mock_open, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_save_path.side_effect = lambda x: f"dummy_path_{x}"
        self.storage._metadata = {"key": "value"}
        self.storage._current_elements = 3

        await self.storage._insert_done()

        self.storage._index.save_index.assert_called_with("dummy_path_hnsw_index_128.bin")
        mock_open.assert_called_with("dummy_path_hnsw_metadata.pkl", "wb")
        mock_logger.debug.assert_called_with("Saving 3 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps(({"key": "value"}, 3)))
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_query_start_with_existing_files(self, mock_logger, mock_open, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.side_effect = lambda x: f"dummy_path_{x}"

        await self.storage._query_start()

        mock_index.assert_called_with(space="cosine", dim=self.storage.embedding_dim)
        self.storage._index.load_index.assert_called_with(
            "dummy_path_hnsw_index_128.bin", max_elements=self.config.max_elements
        )
        self.assertEqual(self.storage._metadata, {"key": "value"})
        self.assertEqual(self.storage._current_elements, 3)
        mock_logger.debug.assert_called_with("Loaded 3 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch("fast_graphrag._storage._vdb_hnswlib.hnswlib.Index")
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_query_start_with_no_files(self, mock_logger, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None

        await self.storage._query_start()

        mock_index.assert_called_with(space="cosine", dim=self.storage.embedding_dim)
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage._current_elements, 0)
        mock_logger.warning.assert_called_with(
            "No data file found for vectordb storage 'None'. Loading empty vectordb."
        )

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_insert_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f"dummy_path_{x}"
        with self.assertRaises(InvalidStorageError):
            await self.storage._insert_start()
        mock_logger.error.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._vdb_hnswlib.logger")
    async def test_query_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f"dummy_path_{x}"
        with self.assertRaises(InvalidStorageError):
            await self.storage._query_start()
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
