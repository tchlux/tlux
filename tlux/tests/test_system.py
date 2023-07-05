
def worker(arr):
    # Access and modify the shared array
    shared_arr = np.ctypeslib.as_array(arr)
    shared_arr[0] = 42

def _test_sharedctypes():
    import numpy as np
    from multiprocessing import Process, sharedctypes
    from ctypes import c_double
    from tlux.profiling import mem_use
    print("mem_use(): ", mem_use(), flush=True)
    # Create a large NumPy array
    initial_mem_use = mem_use()
    large_array = np.zeros((1000, 1000))
    new_mem_use = mem_use()
    array_size = new_mem_use - initial_mem_use
    print("array_size: ", array_size, flush=True)
    # Create a shared memory object
    initial_mem_use = mem_use()
    shared_arr = sharedctypes.RawArray(c_double, large_array.flatten())
    new_mem_use = mem_use()
    shared_array_size = new_mem_use - initial_mem_use
    print("shared_array_size: ", shared_array_size, flush=True)
    # Convert the shared memory object to a NumPy array
    initial_mem_use = mem_use()
    shared_np_array = np.frombuffer(shared_arr, dtype=np.float64).reshape(large_array.shape)
    new_mem_use = mem_use()
    np_buffer_size = new_mem_use - initial_mem_use
    print("np_buffer_size: ", np_buffer_size, flush=True)
    # Start a worker process
    initial_mem_use = mem_use()
    p = Process(target=worker, args=(shared_arr,))
    p.start()
    new_mem_use = mem_use()
    process_size = new_mem_use - initial_mem_use
    p.join()
    print("process_size: ", process_size, flush=True)
    # Check if the shared array was modified
    assert (shared_np_array[0,0] == 42)
    assert (shared_np_array[0,1:].sum() == 0.0)


class SlowObject:
    value = 0.1
    def wait(self, value=None):
        if value is None: value = self.value
        import time
        time.sleep(value)
        return value


def _test_asynchronous(verbose=False, error_tolerance = 0.001):
    print(" test Asynchronous..", end="\n" if verbose else "", flush=True)
    # 
    from tlux.system import Asynchronous, Timer
    AsyncSlowObject = Asynchronous(SlowObject, verbose=False)
    t = Timer()
    # Initialization.
    obj = AsyncSlowObject()
    if verbose: print("  type(obj): ", type(obj), t(), flush=True)
    # Try to look up a missing attribute.
    try: obj.does_not_exist
    except AttributeError: pass
    else: raise(RuntimeError(f"Expected 'AttributeError' when looking up 'obj.does_not_exist'."))
    # Attribute lookup.
    t.start()
    value = obj.value
    assert (t() < error_tolerance), f"Object took too long to fetch an attribute, {t()} seconds."
    if verbose: print("  obj.value: ", value, t(), flush=True)
    # Attribute value retrieval.
    t.start()
    value = obj.value.get()
    assert (t() < error_tolerance), f"Object took too long to get an attribute's value, {t()} seconds."
    if verbose: print("  obj.value.get(): ", value, t(), flush=True)
    # Expensive method call.
    t.start()
    value = obj.wait()
    assert (t() < error_tolerance), f"Object took too long to start working on a method, {t()} seconds."
    if verbose: print("  type(obj.wait()): ", type(value), t(), flush=True)
    # Waiting on expensive method results.
    value = value.get()
    assert (value == SlowObject.value), f"Returned waited time \"{value}\" does not match expected default {SlowObject.value}."
    assert (t() >= SlowObject.value-error_tolerance), f"Object recieved result faster than the expected ~{SlowObject.value} seconds, {t()} seconds."
    if verbose: print("  obj.wait().get(): ", value, t(), flush=True)
    # Try triggering an error within the Aynchronous instance.
    value = obj.wait('hello').get()
    assert (type(value) == TypeError), f"Expected to receive a 'TypeError', but got a '{type(value).__name__}'."
    assert (str(value) == "'str' object cannot be interpreted as an integer"), f"Expected a specific error message, but did not receive the expected message."
    # 
    print(" passed.", flush=True)


if __name__ == "__main__":
    # _test_sharedctypes()
    _test_asynchronous()
