import numpy as np
from tlux.math import is_numeric

# Test examples:
x = 5
y = 3.14
z = "hello"
b = np.float32(1)
a = np.uint8(1)

assert(is_numeric(x))   # True
assert(is_numeric(y))   # True
assert(not is_numeric(z))   # False
assert(is_numeric(a))   # False
assert(is_numeric(b))   # False
