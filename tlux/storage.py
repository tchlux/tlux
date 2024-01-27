import multiprocessing
import os
import pickle
import tempfile
import time
from typing import Any, Iterable, Iterator, Optional, Tuple, Union


DEFAULT_REQUEST_DELAY: float = 0.001
DEFAULT_MAX_DELAY: float = 1.0
DEFAULT_MAX_RETRIES: int = 100


# FileLock is used to ensure that a lock is held during read, write, and delete operations.
# 
# Arguments:
#   lock_path (str): The path to the *desired* lock file (must *not* exist whenever "unlocked").
#   request_delay (float): The time waited in seconds for initial delay between lock attempts.
#   max_retries (int): The maximum number of retries on locking before raising an error.
# 
class FileLock:
    def __init__(self, lock_path: str, request_delay: float = DEFAULT_REQUEST_DELAY, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.lock_path = lock_path  if (not lock_path.endswith(".lock")) else lock_path
        self.request_delay = request_delay
        self.max_retries = max_retries

    def __enter__(self) -> 'FileLock':
        for retry in range(self.max_retries):
            try:
                os.mkdir(self.lock_path)
                return self
            except OSError as e:
                time.sleep(min(self.request_delay * (2 ** retry), DEFAULT_MAX_DELAY))
        else:
            raise Exception("Max retries reached for acquiring file lock")

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        os.rmdir(self.lock_path)


# KeyValueStore is a class for managing a persistently stored dictionary.
# It uses file-based locking and atomic operations for safe concurrent access.
# 
# Arguments:
#   path (str): The path to the pickle file used for storage.
#   request_delay (float): The time waited in seconds for initial delay between lock attempts.
#   max_retries (int): The maximum number of retries on locking before raising an error.
# 
# Attributes:
#   path (str): The path to the data file.
#   lock_path (str): The path to the lock file.
# 
class KeyValueStore:
    def __init__(self, path: Optional[str] = None, request_delay: float = DEFAULT_REQUEST_DELAY, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        # Get the default path (a temporary file).
        self._is_temporary = False
        if (path is None):
            path = tempfile.NamedTemporaryFile(delete=False).name
            self._is_temporary = True
            self._parent_process_pid = multiprocessing.current_process().pid
        # Set attributes.
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.path = path
        self.lock_path = path + '.lock'
        # Internal attributes.
        self._data = {}
        self._last_modified = 0
        # Initialize the key value store.
        with self._lock():
            self._load()
            self._save()

    def _lock(self) -> 'FileLock':
        return FileLock(self.lock_path, self.request_delay, self.max_retries)

    def _load(self) -> None:
        if (os.path.exists(self.path) and (os.path.getsize(self.path) > 0)):
            last_modified = os.path.getmtime(self.path)
            if (self._last_modified < last_modified):
                with open(self.path, 'rb') as f:
                    self._data = pickle.load(f)
                    self._last_modified = last_modified

    def _save(self) -> None:
        with open(self.path, 'wb') as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Destructor to clean up temporary file if it was created (only by creating processs).
    def __del__(self):
        if (self._is_temporary
            and os.path.exists(self.path)
            and self._parent_process_pid == multiprocessing.current_process().pid):
            os.remove(self.path)

    def __repr__(self) -> str:
        # Get the number of elements.
        with self._lock():
            self._load()
            n = len(self._data)
        # Get the file size.
        try:
            size = os.path.getsize(self.path) 
        except OSError:
            size = '???'
        # Return a representation string.
        return f"{self.__class__.__name__}(path='{self.path}', n={n}, size={size})"

    def __str__(self) -> str:
        with self._lock():
            self._load()
            items_str = ', '.join([f"{repr(key)}: {repr(value)}" for key, value in self._data.items()])
        return f"{self.__class__.__name__}({{{items_str}}})"

    # Compute the number of elements in this store and return the integer.
    def __len__(self) -> int:
        with self._lock():
            self._load()
            return len(self._data)

    # Check if a key is in the key-value store.
    # 
    # Arguments:
    #   key (str): The key to check for in the store.
    # 
    # Returns:
    #   bool: True if the key is in the store, False otherwise.
    # 
    def __contains__(self, key: str) -> bool:
        with self._lock():
            self._load()  # Ensure data is up-to-date
            return key in self._data

    # Retrieve an item from the key-value store.
    # 
    # Arguments:
    #   key (str): The key of the item to retrieve.
    # 
    # Returns:
    #   Any: The value associated with the specified key.
    # 
    # Raises:
    #   KeyError: If the key is not found in the store.
    # 
    def __getitem__(self, key: str) -> Any:
        with self._lock():
            self._load()
            return self._data[key]

    # Set an item in the key-value store.
    # 
    # Arguments:
    #   key (str): The key of the item to set.
    #   value (Any): The value to associate with the key.
    # 
    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock():
            self._load()
            self._data[key] = value
            self._save()

    # Delete an item from the key-value store.
    # 
    # Arguments:
    #   key (str): The key of the item to delete.
    # 
    # Raises:
    #   KeyError: If the key is not found in the store.
    # 
    def __delitem__(self, key: str) -> None:
        with self._lock():
            self._load()
            del self._data[key]
            self._save()

    # Iterate over the keys in the key-value store.
    # 
    # This method allows the KeyValueStore to be iterable, enabling iteration over the keys of the store.
    # 
    # Yields:
    #   Iterator[str]: An iterator over the keys of the store.
    # 
    def __iter__(self) -> Iterator[str]:
        with self._lock():
            self._load()
            return iter(self._data)

    # The "get attribute" operator should retrieve attributes of "self._data" when possible.
    # Forward attribute access to the internal dictionary '_data' when the attribute
    # is not found in the KeyValueStore instance itself.
    # 
    # Argumentss:
    #     attr (str): The name of the attribute to retrieve.
    # 
    # Returns:
    #     Any: The attribute from the internal dictionary '_data'.
    # 
    # Raises:
    #     AttributeError: If the attribute is not found in both the KeyValueStore instance and '_data'.
    # 
    def __getattr__(self, attr):
        # Handle special case where _data or _lock is not yet set
        if attr in ['_data', '_lock']:
            raise AttributeError(f"'KeyValueStore' object has no attribute '{attr}'")
        # Otherwise, check for the attribute in 'self._data'.
        if hasattr(self._data, attr):
            with self._lock():
                self._load()
                return getattr(self._data, attr)
        raise AttributeError(f"'KeyValueStore' object has no attribute '{attr}'")

    # Update items in the key-value store.
    # 
    # This method updates the store with the key-value pairs from another dictionary or iterable of key-value pairs.
    # If a key already exists in the store, its value is updated. If a key does not exist, it's added to the store.
    # 
    # Arguments:
    #   other (Union[dict, Iterable[Tuple[str, Any]]]): A dictionary or iterable of key-value pairs to update the store.
    # 
    def update(self, other: Union[dict, Iterable[Tuple[str, Any]]]) -> None:
        with self._lock():
            self._load()
            self._data.update(other)
            self._save()

    # Clear all items from the key-value store.
    def clear(self) -> None:
        with self._lock():
            self._load()
            self._data.clear()
            self._save()
