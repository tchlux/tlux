#!python 

# Any python code that should be executed (as if __main__) during
# setup should go here to prepare the module on a new computer.

# Try compiling the 'fmath' library.
import sys
_first = sys.path.pop(0)
try:
    import os, fmodpy
    _here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math")
    _fmath_path = os.path.join(_here, "fmath.f90")
    _fmath = fmodpy.fimport(_fmath_path, lapack=True, verbose=False, output_dir=_here)
except Exception as exc:
    print(f"WARNING: 'tlux' encountered exception while trying to compile the 'fmath' library.")
finally:
    sys.path.insert(0, _first)
