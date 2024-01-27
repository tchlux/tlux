import multiprocessing
import os
import unittest

from tlux.storage import KeyValueStore


def modify_store(store, key, value):
    """Function for modifying the store in a separate process."""
    store[key] = value
    return store[key]


class TestKeyValueStore(unittest.TestCase):

    def setUp(self):
        """Create a KeyValueStore instance for each test."""
        self.store = KeyValueStore(path='test_store.pkl')

    def tearDown(self):
        """Cleanup the test file after each test."""
        os.remove('test_store.pkl')

    def test_initialization(self):
        """Test store initialization."""
        self.assertIsInstance(self.store, KeyValueStore)
        self.assertFalse(os.path.exists(self.store.lock_path))

    def test_read_write(self):
        """Test writing and reading data."""
        self.store['test_key'] = 'test_value'
        self.assertEqual(self.store['test_key'], 'test_value')

    def test_deletion(self):
        """Test deletion of data."""
        self.store['test_key'] = 'test_value'
        del self.store['test_key']
        with self.assertRaises(KeyError):
            _ = self.store['test_key']

    def test_clear(self):
        """Test clearing all data."""
        self.store['test_key_1'] = 'test_value'
        self.store['test_key_2'] = 'test_value'
        self.store.clear()
        with self.assertRaises(KeyError):
            _ = self.store['test_key_1']
        with self.assertRaises(KeyError):
            _ = self.store['test_key_2']

    def test_non_existent_key(self):
        """Test accessing a non-existent key."""
        with self.assertRaises(KeyError):
            _ = self.store['non_existent_key']

    def test_locking(self):
        """Test that the lock file is created and removed."""
        with self.store._lock():
            self.assertTrue(os.path.exists(self.store.lock_path))
        self.assertFalse(os.path.exists(self.store.lock_path))

    def test_contains_method(self):
        """Test the __contains__ method for key presence."""
        self.store['exist_key'] = 'value'
        self.assertIn('exist_key', self.store)
        self.assertNotIn('non_exist_key', self.store)

    def test_update_method(self):
        """Test the update method for adding and updating keys."""
        initial_data = {'key1': 'value1', 'key2': 'value2'}
        update_data = {'key2': 'new_value2', 'key3': 'value3'}
        # Set initial data
        for key, value in initial_data.items():
            self.store[key] = value
        # Update data
        self.store.update(update_data)
        # Check if data is updated
        self.assertEqual(self.store['key1'], 'value1')  # Unchanged
        self.assertEqual(self.store['key2'], 'new_value2')  # Updated
        self.assertEqual(self.store['key3'], 'value3')  # Added
        self.assertEqual(len(self.store), 3) # 3 elements

    def test_iterator(self):
        """Test the __iter__ method for iterating over keys."""
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        # Populate the store
        for key, value in data.items():
            self.store[key] = value
        # Test iteration over keys
        keys_from_store = set(key for key in self.store)
        self.assertEqual(keys_from_store, set(data.keys()))
        self.assertEqual(len(self.store), 3) # 3 elements

    def test_getattr(self):
        """Test 'get' operations on KeyValueStore."""
        self.assertIsNone(self.store.get("abc", None))
        self.store["abc"] = "abc"
        self.assertEqual(self.store.get("abc", None), "abc")

    def test_dict_conversion(self):
        """Test conversion of KeyValueStore to a dictionary."""
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        # Populate the store
        for key, value in data.items():
            self.store[key] = value
        # Convert KeyValueStore to dict and compare
        store_as_dict = dict(self.store.items())
        self.assertEqual(store_as_dict, data)

    def test_concurrent_access(self):
        """Test concurrent modification of KeyValueStore."""
        store = KeyValueStore()
        num_processes = 50
        processes = []
        # Create multiplie processes that all try to modify a store passed via serialization.
        for i in range(num_processes):
            key = f'key_{i}'
            value = f'value_{i}'
            process = multiprocessing.Process(target=modify_store, args=(store, key, value))
            processes.append(process)
        # Start multiple processes to modify the store
        for process in processes:
            process.start()
        # Wait for all processes to complete
        for process in processes:
            process.join()
        # Verify that all changes are reflected in the local store
        for i in range(num_processes):
            key = f'key_{i}'
            self.assertEqual(store[key], f'value_{i}')
        # Ensure the temporary store is deleted.
        path = store.path
        del(store)
        self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
