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
    except Exception as exc:
        print(f"WARNING: 'tlux' encountered exception while trying to build compiled libraries.")
        print()
        print(exc)
    finally:
        sys.path.insert(0, _first)


# Build the AXY model (compiled Fortran code).
def build_axy():
    import os, fmodpy
    # Get the directory for the AXY compiled source code.
    _dependencies = ["random.f90", "matrix_operations.f90", "sort_and_select.f90", "axy.f90"]
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "axy")
    _path = os.path.join(_dir, "axy.f90")
    _axy = fmodpy.fimport(
        _path, dependencies=_dependencies, output_dir=_dir,
        blas=True, lapack=True, omp=True, wrap=True, # verbose=True, 
        link_blas="", link_lapack="",
        symbols = [
            ("sgemm_", "blas"),
            ("sgels_", "lapack"),
            ("omp_get_max_threads_", "omp")
        ],
        libraries = [
            _dir, np.__path__[0],
            "/usr/lib",
            "/opt/homebrew/Cellar/openblas",
            "/opt/homebrew/Cellar/libomp",
        ],
    )
    return _axy


# Build the BallTree code.
def build_balltree():
    # from util.approximate.delaunay import delsparse
    import os, fmodpy
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "balltree")
    # Ball tree.
    _dependencies = ["swap.f90", "prune.f90", "fast_select.f90", "fast_sort.f90", "ball_tree.f90"]
    _path = os.path.join(_dir, "ball_tree.f90")
    _balltree = fmodpy.fimport(_path, dependencies=_dependencies, output_dir=_dir, omp=True)
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
    
