# type: ignore
import os
import unittest
from unittest.mock import AsyncMock, patch

from fast_graphrag._exceptions import InvalidStorageUsageError
from fast_graphrag._storage._base import BaseStorage, Namespace


class TestNamespace(unittest.TestCase):
    def tearDown(self):
        if os.path.exists("dummy_dir"):
            os.rmdir("dummy_dir")

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_init_creates_working_dir_when_namespace_is_none(self, mock_makedirs, mock_exists):
        ns = Namespace("dummy_dir")
        mock_exists.assert_called_once_with("dummy_dir")
        mock_makedirs.assert_called_once_with("dummy_dir")
        self.assertIsNone(ns.namespace)
        self.assertEqual(ns.working_dir, "dummy_dir")

    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    def test_init_does_not_create_working_dir_when_namespace_is_provided(self, mock_makedirs, mock_exists):
        ns = Namespace("dummy_dir", "dummy_namespace")
        mock_exists.assert_not_called()
        mock_makedirs.assert_not_called()
        self.assertEqual(ns.namespace, "dummy_namespace")
        self.assertEqual(ns.working_dir, "dummy_dir")

    def test_get_resource_path(self):
        ns = Namespace("dummy_dir", "dummy_namespace")
        resource_path = ns.get_resource_path("resource")
        self.assertEqual(resource_path, os.path.join("dummy_dir", "dummy_namespace_resource"))

    def test_get_resource_path_raises_assertion_error_when_namespace_is_none(self):
        ns = Namespace("dummy_dir")
        with self.assertRaises(AssertionError):
            ns.get_resource_path("resource")

    def test_make_for(self):
        ns = Namespace("dummy_dir", "dummy_namespace")
        new_ns = ns.make_for("new_namespace")
        self.assertEqual(new_ns.working_dir, "dummy_dir")
        self.assertEqual(new_ns.namespace, "new_namespace")

    def test_static_new_method(self):
        ns = Namespace.new("dummy_dir")
        self.assertEqual(ns.working_dir, "dummy_dir")
        self.assertIsNone(ns.namespace)

class TestBaseStorage(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.storage = BaseStorage(config=None)

    @patch.object(BaseStorage, '_insert_start', new_callable=AsyncMock)
    @patch.object(BaseStorage, '_query_done', new_callable=AsyncMock)
    @patch("fast_graphrag._storage._base.logger")
    async def test_insert_start_from_query_mode(self, mock_logger, mock_query_done, mock_insert_start):
        self.storage._mode = "query"
        self.storage._in_progress = True

        await self.storage.insert_start()

        mock_query_done.assert_called_once()
        mock_insert_start.assert_called_once()
        mock_logger.warning.assert_called_once()
        self.assertEqual(self.storage._mode, "insert")
        self.assertTrue(self.storage._in_progress)

    @patch.object(BaseStorage, '_insert_start', new_callable=AsyncMock)
    async def test_insert_start_from_none_mode(self, mock_insert_start):
        self.storage._mode = None
        self.storage._in_progress = False

        await self.storage.insert_start()

        mock_insert_start.assert_called_once()
        self.assertEqual(self.storage._mode, "insert")
        self.assertTrue(self.storage._in_progress)

    @patch.object(BaseStorage, '_insert_done', new_callable=AsyncMock)
    async def test_insert_done_in_insert_mode(self, mock_insert_done):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        await self.storage.insert_done()

        mock_insert_done.assert_called_once()
        self.assertFalse(self.storage._in_progress)

    @patch("fast_graphrag._storage._base.logger")
    async def test_insert_done_in_query_mode(self, mock_logger):
        self.storage._mode = "query"
        self.storage._in_progress = True

        with self.assertRaises(InvalidStorageUsageError):
            await self.storage.insert_done()
        mock_logger.error.assert_called_once()

    @patch.object(BaseStorage, '_query_start', new_callable=AsyncMock)
    @patch.object(BaseStorage, '_insert_done', new_callable=AsyncMock)
    @patch("fast_graphrag._storage._base.logger")
    async def test_query_start_from_insert_mode(self, mock_logger, mock_insert_done, mock_query_start):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        await self.storage.query_start()

        mock_insert_done.assert_called_once()
        mock_query_start.assert_called_once()
        mock_logger.warning.assert_called_once()
        self.assertEqual(self.storage._mode, "query")
        self.assertTrue(self.storage._in_progress)

    @patch.object(BaseStorage, '_query_start', new_callable=AsyncMock)
    async def test_query_start_from_none_mode(self, mock_query_start):
        self.storage._mode = None
        self.storage._in_progress = False

        await self.storage.query_start()

        mock_query_start.assert_called_once()
        self.assertEqual(self.storage._mode, "query")
        self.assertTrue(self.storage._in_progress)

    @patch.object(BaseStorage, '_query_done', new_callable=AsyncMock)
    async def test_query_done_in_query_mode(self, mock_query_done):
        self.storage._mode = "query"
        self.storage._in_progress = True

        await self.storage.query_done()

        mock_query_done.assert_called_once()
        self.assertFalse(self.storage._in_progress)

    @patch("fast_graphrag._storage._base.logger")
    async def test_query_done_in_insert_mode(self, mock_logger):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        with self.assertRaises(InvalidStorageUsageError):
            await self.storage.query_done()
        mock_logger.error.assert_called_once()

if __name__ == "__main__":
    unittest.main()
