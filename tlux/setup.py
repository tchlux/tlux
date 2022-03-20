#!python 

# Any python code that should be executed (as if __main__) during
# setup should go here to prepare the module on a new computer.

# Try compiling the 'fmath' library.
try:
    import os, fmodpy
    _here = os.path.dirname(os.path.abspath(__file__))
    _fmath_path = os.path.join(_here, "fmath.f90")
    _fmath = fmodpy.fimport(_fmath_path, verbose=False, output_dir=_here)
except:
    pass
