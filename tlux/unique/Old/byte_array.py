# Holder for a list of values.
class ByteArray:
    # Initialization
    limit = 20
    type = str
    needs_freeing = True
    def __init__(self, values, pointer=None, n=None):
        # Store the length of the array first.
        if (n is None): n = len(values)
        self.n = ctypes.c_long(n)
        self.indices = []
        self.lengths = []
        # Assume we were provided an iterable of elements to
        #  convert to bytes if there was no "pointer" provided.
        if (pointer is None):
            try: self.type = next(iter(values))
            except StopIteration: pass
            bytes_generator = to_bytes(values)
            # Cast the values into a c_char_p (even though their void_p and not
            #  necessarily null-terminated, we'll get the length later).
            # "itertools.islice" gets the first "n" elements from the generator.
            values = (ctypes.c_void_p * n)(*itertools.islice(bytes_generator, n))
            pointer = ctypes.cast(ctypes.pointer(values), ctypes.c_void_p)
            self.type = next(bytes_generator)
            self.indices = next(bytes_generator)
            self.lengths = next(bytes_generator)
            self.needs_freeing = False
        # Store the "values" and the "pointer".
        self.data = values
        self.pointer = pointer
    def __del__(self):
        if (self.needs_freeing and hasattr(self, "n") and hasattr(self, "values")):
            clib.free_unique(ctypes.byref(self.n), ctypes.byref(self.data))
    # List-like behaviors.
    def __getitem__(self, i):
        if (self.lengths[i] < 0):
            return self.data[i].decode()
        elif (self.type is int):
            return self.type.from_bytes(self.data[i], byteorder="little", signed=True)
        else:
            return np.frombuffer(self.data[i], dtype=self.type, count=1)
    def __len__(self):
        return self.n.value
    def __repr__(self):
        return f"<{type(self).__name__}[{len(self)}] of '{self.type.__name__}' at {self.pointer}>"
    def __str__(self):
        s = "["
        # If this is an object with lots of entries, split in half and fill with "...".
        if (len(self) > self.limit):
            s += ", ".join((
                ", ".join((repr(self[i]) for i in range(min(len(self),self.limit//2)))),
                "...",
                ", ".join((repr(self[i]) for i in range(max(0,len(self)-self.limit//2),len(self)))),
            ))
        # Otherwise this object can be shown in full.
        else:
            s += ", ".join((repr(self[i]) for i in range(len(self))))
        s += "]"
        return s    
    # Numpy-like behaviors.
    def tolist(self):
        return [self[i] for i in range(len(self))]
    @property
    def size(self):
        return len(self)
    @property
    def shape(self):
        return (len(self),)
    @property
    def dtype(self):
        return type(self)

