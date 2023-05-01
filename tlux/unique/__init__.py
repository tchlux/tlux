import ctypes
import os
import itertools
import functools
import math
import multiprocessing
import numpy as np


_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "unique.so")

# Import the C library
try:
    clib = ctypes.cdll.LoadLibrary(_lib_path)
except:
    from tlux.setup import build_unique
    clib = build_unique()


# Create a custom subclass of `c_void_p` that keeps a reference to the
#  source array and ensures that it is not deleted from memory.
class ArrayVoidP(ctypes.c_void_p):
    def __init__(self, array, *args, **kwargs):
        self._source_array = array
        super().__init__(*args, **kwargs)


# Function for converting arrays into a C compatible type.
# Inputs:
#   array - Either an np.ndarray, or any python iterable with a __len__. For an
#           ndarray of any fixed-width type, will be converted with the builtin
#           'tobytes'. For arrays of objects or generic iterables, all data will
#           be treated as pythone 'str' objects.
# Outputs:
#   array - The source 'array' object or None, depending on if a reference is needed.
#   voidp - A ctypes.c_void_p object that points to byte
#           representations of the contents of 'array'
#   length - The ctypes.c_long number of elements in the array 'voidp'.
#   width - The ctypes.c_int bytes per data object referenced in 'voidp' if
#           the data type has fixed width, otherwise the value -1.
#   info - A ctypes.c_char_p that contains the name of the data type stored in 'voidp'.
def to_bytes(array):
    # If this is a numpy array, then flatten it first (to iterate over elements).
    if (isinstance(array, np.ndarray)):
        array = array.flatten()
    # If this is a numpy array and it is not contiguous, reallocate.
    if (isinstance(array, np.ndarray) and (not array.data.c_contiguous)):
        array = np.asarray(array, order="C")
    # If this is a numpy array, 
    if (isinstance(array, np.ndarray) 
        and (array.dtype.name != 'object')):
        # Convert it directly into the correctly sized C array.
        voidp = ArrayVoidP(array, array.ctypes.data)
        length = ctypes.c_long(len(array))
        width = ctypes.c_int(array.dtype.itemsize)
        info = ctypes.c_char_p(array.dtype.name.encode())
    else:
        # This is a structure of python objects, convert everything to strings, encode
        #  those, put bytes in an array of ctypes.c_char_p, cast that to c_void_p.
        charp = (ctypes.c_char_p * len(array))(*(str(i).encode() for i in array))
        voidp = ctypes.cast(charp, ctypes.c_void_p)
        length = ctypes.c_long(len(array))
        width = ctypes.c_int(-1)
        info = ctypes.c_char_p(b'object')
    # Return the final values (be sure to keep a reference to the 'array' so it isn't freed).
    return voidp, length, width, info


# Function for converting objects created with 'to_bytes' back into the
# source object. Inputs:
#   voidp - A ctypes.c_void_p object that points to an array of data.
#   length - The ctypes.c_long number of elements in the array 'voidp'.
#   width - The ctypes.c_int bytes per data object referenced in 'voidp' if the
#           data type has fixed width, otherwise the value -1.
#   info - A ctypes.c_char_p that contains the name of the data type stored in 'voidp'.
# 
# Outputs:
#   array - A np.ndarray object containing the appropriately formed data (either the
#           correct fixed-width numbers, or python 'str' objects for strings).
def from_bytes(voidp, length, width, info):
    # Appropriately load the data from bytes.
    if (width.value == -1):
        # If the width is -1, the data is of the 'str' type.
        # Extract the bytes and decode them into 'str' objects.
        charp = (ctypes.c_char_p * length.value).from_address(voidp.value)
        # Convert the list of 'str' objects into a numpy array.
        array = np.fromiter((s for s in charp), dtype=object)
    else:
        # If the width is not -1, the data is of a fixed-width type.
        # Create a numpy array of the appropriate data type from the raw bytes.
        voidp = (ctypes.c_void_p * length.value * width.value).from_address(voidp.value)
        array = np.frombuffer(voidp, dtype=info.value.decode(), count=length.value)
    return array


# Holder for array data (that may have been created with custom C library).
#  Includes functionality to make it behave somewhat like a NumPy array.
class ByteArray:
    def __init__(self, array, voidp, length, width, info, needs_freeing, print_limit=20):
        self.array = array
        self.voidp = voidp
        self.length = length
        self.width = width
        self.info = info
        self.needs_freeing = needs_freeing
        self.print_limit = print_limit
    # Cleanup for C-allocated data.
    def __del__(self):
        if (hasattr(self, "needs_freeing") and (self.needs_freeing)):
            clib.free_unique(
                ctypes.byref(self.voidp),
                ctypes.byref(self.length),
                ctypes.byref(self.width)
            )
    # List-like behaviors.
    def __getitem__(self, i):
        if (self.width.value == -1):
            return self.array[i].decode()
        else:
            return self.array[i].item()
    def __len__(self):
        return self.array.size
    def __repr__(self):
        return f"<{type(self).__name__}[{len(self)}] of '{self.info.value.decode()}' at {ctypes.cast(self.voidp, ctypes.c_void_p).value}>"
    def __str__(self):
        s = "["
        # If this is an object with lots of entries, split in half and fill with "...".
        if (len(self) > self.print_limit):
            s += " ".join((
                " ".join((repr(self[i]) for i in range(min(len(self),self.print_limit//2)))),
                "...",
                " ".join((repr(self[i]) for i in range(max(0,len(self)-self.print_limit//2),len(self)))),
            ))
        # Otherwise this object can be shown in full.
        else:
            s += " ".join((repr(self[i]) for i in range(len(self))))
        s += "]"
        return s    
    # Numpy-like behaviors.
    @property
    def size(self):
        return len(self)
    @property
    def shape(self):
        return (len(self),)
    @property
    def dtype(self):
        return type(self)
    # WARNING: This 'tolist' implementation includes metadata, designed only to be consumed by 'fromlist' of this class.
    def tolist(self):
        return [
            [self[i] for i in range(len(self))],
            self.length.value,
            self.width.value,
            self.info.value.decode(),
        ]
    # Method for loading a value created with 'tolist'.
    @staticmethod
    def fromlist(list_value):
        # Unpack expected items from list.
        array, length, width, info = list_value
        # Convert the array into the expected type.
        array = np.asarray(array, dtype=info)
        # Convert the array (of the expected type) into C-compatible bytes.
        voidp, length, width, info = to_bytes(array)
        array = from_bytes(voidp, length, width, info)
        # For types that are not strings, we need to convert 'voidp' into an array of pointers.
        if (width.value != -1):
            voidp = (ctypes.c_void_p * length.value)(*(
                ctypes.c_void_p(voidp.value + i*width.value)
                for i in range(length.value)
            ))
        return ByteArray(array, voidp, length, width, info, needs_freeing=False)


# Return the sorted unique elements in an array.
def unique(array):
    # Create a C compatible version of this array.
    voidp, length, width, info = to_bytes(array)
    temp = from_bytes(voidp, length, width, info)
    # return sorted(set(map(str,array)))
    # 
    # Call the parallel sort function.
    num_unique = ctypes.c_long()
    sorted_unique_pointer = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))()
    clib.unique(voidp, length, width, ctypes.byref(num_unique), ctypes.byref(sorted_unique_pointer))
    # Check for errors from the 'unique' routine.
    if (num_unique.value == -1):
        raise(TypeError(f"The 'unique' C library cannot handle items of byte-width {width.value} (type '{info.value.decode}'). Consider converting to python strings or requesting support via a GitHub Issue."))
    # Get the pointer to the raw array of unique values (contiguously packed).
    sorted_unique = ctypes.cast(
        sorted_unique_pointer if (width.value == -1) else sorted_unique_pointer.contents,
        ctypes.c_void_p
    )
    # Transfer the data over to an object that behaves like a list or numpy array,
    # while also including code to free the memory allocated internally when it is deleted.
    unique_array = from_bytes(sorted_unique, num_unique, width, info)
    return ByteArray(unique_array, sorted_unique_pointer, num_unique, width, info, needs_freeing=True)


# Given an array and a mapping that is the result of calling "unique",
#  return an array of integers where all elements of "array" hvae been,
#  replaced with either their reference integer in mapping or 0 for unknown.
def to_int(array, mapping):
    array = np.asarray(array, dtype=mapping.array.dtype).flatten()
    num_array = np.zeros(array.shape, dtype="long")
    # return num_array
    # 
    # Get the C-compatible representation of the data in 'array'.
    voidp, length, width, info = to_bytes(array)
    clib.to_int(
        # Configuration.
        mapping.width,
        # Input data.
        length,
        voidp,
        # Mapping.
        mapping.length,
        ctypes.cast(mapping.voidp, ctypes.c_void_p),
        # Output.
        ctypes.c_void_p(num_array.ctypes.data),
    )
    return num_array



# Verify some basic behavior of the 'to_bytes' and 'from_bytes' functions.
def _test_to_bytes_from_bytes():
    #   Python list.
    a = ['a', 'bcd', None, 1, 3.04, 'êƒ≈ß ', [1,2.0,"-3"], 'bcd', [1,2.0,"-3"], None]
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (tuple(map(str,a)) == tuple(map(lambda v: v.decode(), b))), f"\n{a}\n{b}"
    #   Numpy array of objects.
    a = np.asarray([['a', 'bcd', None], [1, 3.04, 'êƒ≈ß '], [1, 2.0, "-3"]], dtype=object)
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (tuple(map(str,a.flatten())) == tuple(map(lambda v: v.decode(), b))), f"\n{a}\n{b}"
    #   Numpy array of uint8.
    a = np.asarray([0, 4, 30, 255, 30, 4, 0], dtype="uint8")
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (tuple(a.flatten()) == tuple(b)), f"\n{a}\n{b}"
    #   Numpy array of int32.
    a = np.asarray([0, 4, 30, 255, 30, 4, 0], dtype="int32")
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (tuple(a.flatten()) == tuple(b)), f"\n{a}\n{b}"
    #   Numpy array of int64.
    a = np.asarray([0, 4, 30, 255, 30, 4, 0], dtype="int64") * 2**55
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (tuple(a.flatten()) == tuple(b)), f"\n{a}\n{b}"
    #   Numpy array of float32.
    a = np.asarray([float('inf'), 4.222, 30.178, -255.1830, float('nan'), 4.222, float('inf'), float('nan')], dtype="float32")
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (str(tuple(a.flatten())) == str(tuple(b))), f"\n{a}\n{b}"
    #   Numpy array of float64.
    a = np.asarray([float('inf'), 4.222, 30.178, -255.1830, float('nan'), 4.222, float('inf'), float('nan')], dtype="float64")
    output = to_bytes(a)
    b = from_bytes(*output)
    assert (str(tuple(a.flatten())) == str(tuple(b))), f"\n{a}\n{b}"

# Verify behavior of the 'unique' function.
def _test_unique():
    # Run a test, raise an error if it failes.
    def _run_test(a, test, fromlist=False):
        # Use the provided functions to generate a guess.
        u = unique(a)
        if fromlist: u = ByteArray.fromlist(u.tolist())
        i = to_int(test, u)
        # Generate a "ground truth" answer.
        if (u.info.value == b"object"):
            su = sorted(set(map(str, a)), key=lambda v: (len(v), v))
            si = np.asarray([su.index(str(v))+1 if str(v) in su else 0 for v in test])
        else:
            su = sorted(set(a.flatten()))
            si = np.asarray([su.index(v)+1 if v in su else 0 for v in test])
        if (not all(i == si)):
            print("a:    ", a, flush=True)
            print("test: ", test, flush=True)
            print()
            print("unique(a):    ", u, flush=True)
            print("to_int(a, u): ", i, flush=True)
            print()
            print("unique(a):    ", su, flush=True)
            print("to_int(a, u): ", si, flush=True)
            print()
            raise(ValueError(f"Expected the integer mappings to be the same."))

    np.random.seed(0)

    # Test with python list of objects.
    a = ['a', 'bcd', None, 1, 3.04, 'êƒ≈ß ', [1,2.0,"-3"], 'bcd', [1,2.0,"-3"], None]
    test = [a[3], a[6], -100.0] + a[1::2] + [10]
    _run_test(a, test)
    _run_test(a, test, fromlist=True)

    # Test with numpy array of uint8.
    a = np.arange(0, 256, 16).astype("uint8")
    np.random.shuffle(a)
    a = a[:8]
    test = np.arange(0, 256).astype("uint")
    _run_test(a, test)
    _run_test(a, test, fromlist=True)

    # Test with numpy array of ints
    a = np.arange(-2**30, 2**30, 2**27).astype("int32")
    np.random.shuffle(a)
    a = a[:8]
    test = np.arange(0, 128).astype("int32")
    _run_test(a, test)
    _run_test(a, test, fromlist=True)

    # Test with numpy array of longs
    a = np.arange(-2**61, 2**61, 2**58).astype("int64")
    np.random.shuffle(a)
    a = a[:8]
    test = np.arange(0, 128).astype("int64")
    _run_test(a, test)
    _run_test(a, test, fromlist=True)



# For testing purposes.
if __name__ == "__main__":
    # Run logic tests.
    _test_to_bytes_from_bytes()
    _test_unique()

    # Make sure we can force garbage collection.
    import gc
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

    assert (tuple(i.tolist()) == tuple(truth.flatten().tolist())), f"ERROR: C-Library generated values do not match truth.\n  clib: {i.tolist()}\n truth: {truth.tolist()}\n"

    # Show initial profiles.
    if (hasattr(unique, "show_profile")): unique.show_profile()
    if (hasattr(to_int, "show_profile")): to_int.show_profile()
    print(f"mem_use: {mem_use():.2f}MB", flush=True)

    # Call the functions repeatedly and look for a memory leak.
    # @profile
    def testing(steps=100000):
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

