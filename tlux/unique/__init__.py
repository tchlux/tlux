import ctypes
import gc
import os
import numpy as np


_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "unique.so")

# Import the C library
try:
    clib = ctypes.cdll.LoadLibrary(_lib_path)
except:
    from tlux.setup import build_unique
    clib = build_unique()


# Holder for a list of values.
class StringArray:
    # Initialization
    limit = 20
    encoding = "UTF8"
    needs_freeing = True
    def __init__(self, values, pointer=None):
        self.n = ctypes.c_long(len(values))
        # Assume we were provided a list of strings, convert to the appropriate C type.
        if (pointer is None):
            values = (ctypes.c_char_p * len(values))(*(s.encode(self.encoding) for s in values))
            pointer = ctypes.cast(ctypes.pointer(values), ctypes.c_void_p)
            self.needs_freeing = False
        self.values = values
        self.pointer = pointer
    def __del__(self):
        if (self.needs_freeing):
            clib.free_unique(ctypes.byref(self.n), ctypes.byref(self.values))
    # List-like behaviors.
    def __getitem__(self, i):
        return self.values[i].decode(self.encoding)
    def __len__(self):
        return self.n.value
    def __repr__(self):
        return f"<{type(self).__name__}[{len(self)}] at {self.pointer}>"
    def __str__(self):
        s = "["
        # If this is an object with lots of entries, split in half and fill with "...".
        if (len(self) > self.limit):
            s += ", ".join((
                ", ".join((repr(v.decode(self.encoding)) for i,v in enumerate(self.values) if i < self.limit//2)),
                "...",
                ", ".join((repr(v.decode(self.encoding)) for i,v in enumerate(self.values) if (len(self)-i) <= self.limit//2))
            ))
        # Otherwise this object can be shown in full.
        else:
            s += ", ".join((repr(v.decode(self.encoding)) for i,v in enumerate(self.values)))
        s += "]"
        return s    
    # Numpy-like behaviors.
    def shape(self):
        return (len(self),)
    def tolist(self):
        return [s.decode(self.encoding) for s in self.values]
    @property
    def dtype(self):
        return type(self)


# Return the sorted unique elements in an array.
def unique(array):
    # Create a numpy array of objects
    array = np.asarray(array, dtype=object).flatten()
    # return sorted(set(map(str,array)))
    # 
    # Call the parallel sort function.
    num_unique = ctypes.c_long()
    sorted_unique = ctypes.c_void_p()
    clib.unique(ctypes.c_long(array.size), ctypes.c_void_p(array.ctypes.data),
                ctypes.byref(num_unique), ctypes.byref(sorted_unique))
    # Extract the unique elements, converting from C char** to a list of python strings.
    c_unique = (ctypes.c_char_p * num_unique.value).from_address(sorted_unique.value)
    # Transfer the data over to an object that behaves like a list or numpy array,
    # while also including code to free the memory allocated internally when it is deleted.
    return StringArray(values=c_unique, pointer=sorted_unique)


# Given an array and a mapping that is the result of calling "unique",
#  return an array of integers where all elements of "array" hvae been,
#  replaced with either their reference integer in mapping or 0 for unknown.
def to_int(array, mapping):
    num_array = np.zeros(array.shape, dtype="long")
    # return num_array
    # 
    array = np.asarray(array, dtype=object).flatten()
    clib.to_int(
        ctypes.c_long(array.size),
        ctypes.c_void_p(array.ctypes.data),
        ctypes.c_long(len(mapping)),
        ctypes.byref(mapping.pointer),
        ctypes.c_void_p(num_array.ctypes.data),
    )
    return num_array


# For testing purposes.
if __name__ == "__main__":
    # Ensure that profiling is enabled.
    from tlux.profiling import profile, inspect_memory, mem_use
    profiling = False
    if (profiling and (not hasattr(unique, "show_profile"))):
        unique = profile(unique)
    if (profiling and (not hasattr(to_int, "show_profile"))):
        to_int = profile(to_int)

    # Define a test array.
    array = np.asarray([['abc', 'de', 'f',  None, 'ghijk', 'f',  'de'],
                        [unique, 10,   1,   3.0,  [0,1],   (0,1), 10]], dtype=object)

    inspect_memory()

    # Test "unique".
    u = unique(array)
    print()
    print(u)
    truth = sorted(
        set(map(str, array.flatten().tolist())),
        key=lambda i: (len(i), i)
    )
    print(truth)
    print()

    assert (tuple(u) == tuple(truth)), f"ERROR: C-Library generated values do not match truth.\n  clib: {u}\n truth: {truth}\n"

    # Test "to_int".
    print(array)
    array[0,1] = 'NOPE' # to_int
    i = to_int(array, u)
    print()
    print(i)
    truth = np.asarray(
        [truth.index(str(v))+1 if str(v) in truth else 0
         for v in array.flatten()],
        dtype=int
    ).reshape(array.shape)
    print(truth)
    print()

    assert (tuple(i.tolist()) == tuple(truth.tolist())), f"ERROR: C-Library generated values do not match truth.\n  clib: {i.tolist()}\n truth: {truth.tolist()}\n"

    # Show initial profiles.
    if (hasattr(unique, "show_profile")): unique.show_profile()
    if (hasattr(to_int, "show_profile")): to_int.show_profile()
    print(f"mem_use: {mem_use():.2f}MB", flush=True)

    # Call the functions repeatedly and look for a memory leak.
    # @profile
    def testing(steps=10000):
        print("_"*70)
        print(f"Calling {steps} times..")
        for i in range(steps):
            mapping = unique(array)
            i_array = to_int(array, mapping)
            del mapping
            del i_array
        print("^"*70)
        import gc
        gc.collect()
    if (profiling and (not hasattr(testing, "show_profile"))):
        testing = profile(testing)

    if (profiling): inspect_memory()
    for i in range(5):
        testing()
        print(f"mem_use: {mem_use():.2f}MB", flush=True)
        if (hasattr(testing, "show_profile")): testing.show_profile()
        if (profiling): inspect_memory()
