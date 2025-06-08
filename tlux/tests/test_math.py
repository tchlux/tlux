import numpy as np
from tlux.math import is_numeric

if __name__ == "__main__":
    # Test examples:
    x = 5
    y = 3.14
    z = "hello"
    b = np.float32(1)
    a = np.uint8(1)

    assert(is_numeric(x))
    assert(is_numeric(y))
    assert(not is_numeric(z))
    assert(is_numeric(a))
    assert(is_numeric(b))
