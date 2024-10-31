# type: ignore
import pickle
import unittest
from unittest.mock import MagicMock, mock_open, patch

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._storage._blob_pickle import PickleBlobStorage


class TestPickleBlobStorage(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.namespace = MagicMock()
        self.namespace.get_resource_path.return_value = "blob_data.pkl"
        self.storage = PickleBlobStorage(namespace=self.namespace, config=None)

    async def test_get(self):
        self.storage._data = {"key": "value"}
        result = await self.storage.get()
        self.assertEqual(result, {"key": "value"})

    async def test_set(self):
        blob = {"key": "value"}
        await self.storage.set(blob)
        self.assertEqual(self.storage._data, blob)

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps({"key": "value"}))
    async def test_insert_start_with_existing_file(self, mock_open, mock_exists):
        await self.storage._insert_start()
        self.assertEqual(self.storage._data, {"key": "value"})
        mock_open.assert_called_once_with("blob_data.pkl", "rb")

    @patch("os.path.exists", return_value=False)
    async def test_insert_start_without_existing_file(self, mock_exists):
        await self.storage._insert_start()
        self.assertIsNone(self.storage._data)

    @patch("builtins.open", new_callable=mock_open)
    async def test_insert_done(self, mock_open):
        self.storage._data = {"key": "value"}
        await self.storage._insert_done()
        mock_open.assert_called_once_with("blob_data.pkl", "wb")
        mock_open().write.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps({"key": "value"}))
    async def test_query_start_with_existing_file(self, mock_open, mock_exists):
        await self.storage._query_start()
        self.assertEqual(self.storage._data, {"key": "value"})
        mock_open.assert_called_once_with("blob_data.pkl", "rb")

    @patch("os.path.exists", return_value=False)
    @patch("fast_graphrag._storage._blob_pickle.logger")
    async def test_query_start_without_existing_file(self, mock_logger, mock_exists):
        await self.storage._query_start()
        self.assertIsNone(self.storage._data)
        mock_logger.warning.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._blob_pickle.logger")
    async def test_insert_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f"dummy_path_{x}"
        with self.assertRaises(InvalidStorageError):
            await self.storage._insert_start()
        mock_logger.error.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._blob_pickle.logger")
    async def test_query_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f"dummy_path_{x}"
        with self.assertRaises(InvalidStorageError):
            await self.storage._query_start()
        mock_logger.error.assert_called_once()

    async def test_query_done(self):
        await self.storage._query_done()  # Should not raise any exceptions


if __name__ == "__main__":
    unittest.main()
