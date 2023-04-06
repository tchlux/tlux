#!python 

# Any python code that should be executed (as if __main__) during
# setup should go here to prepare the module on a new computer.

import numpy as np

# Function to run during module setup.
def setup():
    # Try compiling the 'fmath' library.
    import sys
    _first = sys.path.pop(0)
    try:
        build_axy()
        build_balltree()
        build_delaunay()
        build_regex()
    except Exception as exc:
        print(f"WARNING: 'tlux' encountered exception while trying to build compiled libraries.")
        print()
        print(exc)
    finally:
        sys.path.insert(0, _first)


# Build the AXY model (compiled Fortran code).
def build_axy():
    import os, fmodpy
    # Define the output directory.
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "axy")
    # Import the RANDOM module, it is self contained and simple.
    _random = fmodpy.fimport(
        input_fortran_file = os.path.join(_dir, "random.f90"),
        output_dir = _dir
    )
    # Import the AXY module, it has many dependencies and uses BLAS and LAPACK libraries.
    #   Get the directory for the AXY compiled source code.
    _dependencies = ["axy_random.f90", "axy_matrix_operations.f90", "axy_sort_and_select.f90", "axy.f90"]
    _path = os.path.join(_dir, "axy.f90")
    _axy_kwargs = dict(
        input_fortran_file = _path,
        dependencies = _dependencies,
        output_dir = _dir,
        blas = True,
        lapack = True,
        omp = True,
        wrap = True,
        verbose = False,
    )
    #   Try and build using (local) defaults for fmodpy compilation.
    try:
        _axy = fmodpy.fimport(**_axy_kwargs)
        _ = _axy.AXY()
    #   When local default fail, try specifying exactly what shared library functions are needed.
    except:
        _axy = fmodpy.fimport( **_axy_kwargs,
            link_blas="", link_lapack="", rebuild=True,
            libraries = [_dir] + fmodpy.config.libraries,
            symbols = [
                ("sgemm", "blas"),
                ("sgels", "lapack"),
                ("omp_get_max_threads", "omp")
            ],
        )
    # Return the AXY and RANDOM modules.
    return _axy, _random


# Build the BallTree code.
def build_balltree():
    # from util.approximate.delaunay import delsparse
    import os, fmodpy
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "balltree")
    # Ball tree.
    _dependencies = ["swap.f90", "prune.f90", "fast_select.f90", "fast_sort.f90", "ball_tree.f90"]
    _path = os.path.join(_dir, "ball_tree.f90")
    _balltree = fmodpy.fimport(
        _path, dependencies=_dependencies, output_dir=_dir, omp=True,
        libraries = [_dir] + fmodpy.config.libraries,
        symbols = [("omp_get_max_threads", "omp")],
    )
    # Fast sort.
    _dependencies = ["swap.f90", "fast_sort.f90"]
    _path = os.path.join(_dir, "fast_sort.f90")
    _fast_sort = fmodpy.fimport(_path, dependencies=_dependencies, output_dir=_dir)
    # Fast select.
    _dependencies = ["swap.f90", "fast_select.f90"]
    _path = os.path.join(_dir, "fast_select.f90")
    _fast_select = fmodpy.fimport(_path, dependencies=_dependencies, output_dir=_dir)
    # Prune.
    _path = os.path.join(_dir, "prune.f90")
    _prune = fmodpy.fimport(_path, output_dir=_dir)
    return _balltree.ball_tree, _fast_sort.fast_sort, _fast_select.fast_select, _prune.prune


# Build the Delaunay code.
def build_delaunay():
    # from util.approximate.delaunay import delsparse
    import os, fmodpy
    _dependencies = ["lapack.f", "slatec.f", "delsparse.f90"]
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "delaunay")
    _path = os.path.join(_dir, "delsparse.f90")
    _delsparse = fmodpy.fimport(
        _path, dependencies=_dependencies, output_dir=_dir,
        omp=True, f_compiler_args="-std=legacy -fPIC -shared -O3",
    )
    return _delsparse
    

# Build the regex library.
def build_regex():
    import os, ctypes
    # Configure for the compilation for the C code.
    _c_compiler = "cc"
    _clib_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regex", "regex.c")
    _clib_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regex", "libregex.so")
    try:
        # Import existing compiled library.
        _regex = ctypes.CDLL(_clib_bin)
    except:
        # Compile and import.
        _compile_command = f"{_c_compiler} -O3 -fPIC -shared -o '{_clib_bin}' '{_clib_source}'"
        os.system(_compile_command)
        _regex = ctypes.CDLL(_clib_bin)
    # Return the compiled module.
    return _regex
