import logging
import multiprocessing
import os
import pickle
import tempfile
import time
from typing import Any, Iterable, Iterator, Optional, Tuple, Union


HOST: str = os.uname().nodename  # Get the current host.
USER: str = os.getlogin()  # Get the current user.
PID: str = str(os.getpid())  # Get the current process ID.
LOCK_FILE_SEPERATOR: str = "--LOCK--"
DEFAULT_REQUEST_DELAY: float = 0.001
DEFAULT_MAX_DELAY: float = 1.0
DEFAULT_MAX_RETRIES: int = 100
DEFAULT_LOCK_DURATION_SEC: int = 10
DEFAULT_WRITE_DELAY_SEC: float = 0.0


# Check if a process with the given PID is currently active.
def is_pid_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# FileLock is used to ensure that a lock is held during read, write, and delete operations.
# 
# Arguments:
#   lock_path (str): The path to the *desired* lock file (must *not* exist whenever "unlocked").
#   lock_duration (int): The number of seconds for which competing operations should allow the lock to be held before ignoring it.
#   request_delay (float): The time waited in seconds for initial delay between lock attempts.
#   max_retries (int): The maximum number of retries on locking before raising an error.
# 
class FileLock:
    class LockFailure(Exception): pass

    def __init__(self, lock_path: str, lock_duration: int = DEFAULT_LOCK_DURATION_SEC, request_delay: float = DEFAULT_REQUEST_DELAY, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.lock_path: str = lock_path if (not lock_path.endswith(".lock")) else lock_path
        self.lock_duration: int = lock_duration
        self.lock_info_path: str = os.path.join(self.lock_path, LOCK_FILE_SEPERATOR.join([HOST, USER, PID, str(self.lock_duration)]))
        self.default_lock_info: str = LOCK_FILE_SEPERATOR.join([HOST, USER, PID, str(self.lock_duration)])
        self.request_delay: float = request_delay
        self.max_retries: int = max_retries
        self.acquisition_time: float = 0.0

    def __enter__(self) -> 'FileLock':
        for retry in range(self.max_retries):
            try:
                os.mkdir(self.lock_path)
                os.mkdir(self.lock_info_path)
                self.acquisition_time = os.path.getctime(self.lock_info_path)
                return self
            except OSError:
                # Check the file hostname+user+PID in the directory, release if the host and user
                #  is same as this process, and that PID does not exist (has terminated).
                try: lock_info: str = next(iter(os.listdir(self.lock_path)))
                except (StopIteration, FileNotFoundError): lock_info: str = self.default_lock_info
                host, user, pid, lock_duration = lock_info.split(LOCK_FILE_SEPERATOR)
                # Try to get the creation time of the lock, if it's gone, then cycle a retry.
                try:
                    lock_expiration = os.path.getctime(self.lock_path) + float(lock_duration)
                except FileNotFoundError:
                    continue
                # Check for lock expiration conditions.
                if time.time() >= lock_expiration:
                    # WARNING: Lock failures and race conditions are possible due to the below operation.
                    try:
                        logging.warning(f"Forcibly removing lock {repr(self.lock_path)}..")
                        os.rmdir(self.lock_info_path)
                        os.rmdir(self.lock_path)
                    except (FileNotFoundError, OSError): pass
                    continue
                elif (((host, user) == (HOST, USER)) and (not is_pid_active(int(pid)))):
                    try:
                        logging.warning(f"Removing lock at {repr(self.lock_path)} held by nonliving process {pid}..")
                        os.rmdir(self.lock_info_path)
                        os.rmdir(self.lock_path)
                    except (FileNotFoundError, OSError): pass
                    continue
                else:
                    time.sleep(min(self.request_delay * (2 ** retry), DEFAULT_MAX_DELAY))
        else:
            raise FileLock.LockFailure("Max retries reached for acquiring file lock")

    # Arguments (unused) are:
    #   exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]
    def __exit__(self, *_) -> None:
        # Attempt to gracefully exit.
        try:
            if (os.path.getctime(self.lock_info_path) != self.acquisition_time):
                raise FileLock.LockFailure(f"Lock acquisition time does not match expected value. Race condition encountered.")
        except Exception:
            raise FileLock.LockFailure(f"Lock changed during execution. Race condition encountered.")
        try:
            os.rmdir(self.lock_info_path)
            os.rmdir(self.lock_path)
        except:
            raise FileLock.LockFailure(f"Lock removed during execution. Race condition encountered.")


# KeyValueStore is a class for managing a persistently stored dictionary.
# It uses file-based locking and atomic operations for safe concurrent access.
# 
# Arguments:
#   path (str): The path to the pickle file used for storage.
#   request_delay (float): The time waited in seconds for initial delay between lock attempts.
#   max_retries (int): The maximum number of retries on locking before raising an error.
#   write_delay_sec (float): The minimum number of seconds elapsed between writes.
# 
# Attributes:
#   path (str): The path to the data file.
#   lock_path (str): The path to the lock file.
# 
class KeyValueStore:
    def __init__(self, path: Optional[str] = None, request_delay: float = DEFAULT_REQUEST_DELAY, max_retries: int = DEFAULT_MAX_RETRIES, write_delay_sec: float = DEFAULT_WRITE_DELAY_SEC) -> None:
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
        self.write_delay_sec = write_delay_sec
        # Internal attributes.
        self._data = {}
        self._last_modified = 0
        self._last_write = 0.0
        # Initialize the key value store.
        with self._lock():
            self._load()
            self._save()

    def _lock(self) -> 'FileLock':
        return FileLock(lock_path=self.lock_path, request_delay=self.request_delay, max_retries=self.max_retries)

    def _load(self) -> None:
        if (os.path.exists(self.path) and (os.path.getsize(self.path) > 0)):
            last_modified = os.path.getmtime(self.path)
            if (self._last_modified < last_modified):
                with open(self.path, 'rb') as f:
                    self._data = pickle.load(f)
                    self._last_modified = last_modified

    def _save(self) -> None:
        delay = time.time() - self._last_write
        if (delay <= self.write_delay_sec): return
        # Write the local file.
        with open(self.path, 'wb') as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Store the time of completing the last write.
        self._last_write = time.time()

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
