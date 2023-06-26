#!python 

# Any python code that should be executed (as if __main__) during
# setup should go here to prepare the module on a new computer.


# Build the unique library.
def build_unique():
    import os, ctypes, fmodpy
    # Configure the compilation.
    _verbose = True
    # _c_compiler = "cc"  # <- on macOS with Clang, no nested functions allowed.
    _c_compiler = fmodpy.config.f_compiler  # <- works with gfortran, assuming true for others too.
    _flags = "-shared -fPIC -O3"  # <- shared library, position independent, O3 optimization.
    # _flags += " -D_UNIQUE_DEBUG_ENABLED"  # <- enable for testing / debugging.
    _flags_by_safety = [
        "-fopenmp -D_UNIQUE_ALLOW_LOCAL_FUNCTIONS", # <- slower for small cases, good for *very* large data.
        "-D_UNIQUE_SINGLE_THREADED -D_UNIQUE_ALLOW_LOCAL_FUNCTIONS", # <- disables OpenMP parallelism and imports.
        "-D_UNIQUE_SINGLE_THREADED", # ^^ enable if compiler supports local functions for variable width comparison.
        # ^^ safest option, single threaded with no local functions.
    ]
    _lib_dir = os.path.dirname(os.path.abspath(__file__))
    _lib_path = os.path.join(_lib_dir, "unique", "unique.so")
    _src_path = os.path.join(_lib_dir, "unique", "unique.c")
    try:
        _unique = ctypes.cdll.LoadLibrary(_lib_path)
    except:
        # Compile the shared library.
        import sys
        # Trial one to compile and load.
        try:
            cmd = f'{_c_compiler} {_flags} {_flags_by_safety[0]} -o "{_lib_path}" "{_src_path}"'
            if _verbose: print("  ", cmd, flush=True)
            os.system(cmd)
            if _verbose: print(flush=True)
            _unique = ctypes.cdll.LoadLibrary(_lib_path)
        except:
            # Trial two to compile and load.
            try:
                cmd = f'{_c_compiler} {_flags} {_flags_by_safety[1]} -o "{_lib_path}" "{_src_path}"'
                if _verbose: print(" ", cmd, flush=True)
                os.system(cmd)
                if _verbose: print(flush=True)
                _unique = ctypes.cdll.LoadLibrary(_lib_path)
            # Trial three to compile and load.
            except:
                cmd = f'{_c_compiler} {_flags} {_flags_by_safety[2]} -o "{_lib_path}" "{_src_path}"'
                if _verbose: print(" ", cmd, flush=True)
                os.system(cmd)
                if _verbose: print(flush=True)
                _unique = ctypes.cdll.LoadLibrary(_lib_path)
    # Return the compiled module.
    return _unique


# Build the AXY model (compiled Fortran code).
def build_axy():
    import os, fmodpy
    # Define the output directory.
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approximate", "axy")
    # Import the RANDOM module, it is self contained and simple.
    _random = fmodpy.fimport(
        input_fortran_file = os.path.join(_dir, "axy_random.f90"),
        dependencies = ["pcg32.f90"],
        output_dir = _dir
    )
    # Import the AXY module, it has many dependencies and uses BLAS and LAPACK libraries.
    #   Get the directory for the AXY compiled source code.
    _dependencies = ["pcg32.f90", "axy_profiler.f90", "axy_random.f90", "axy_matrix_operations.f90", "axy_sort_and_select.f90", "axy.f90"]
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
        # Try calling the underlying library.
        _axy.axy.new_model_config(
            adn=2, ado=3, ads=4, ans=1, ane=6, ade=7,
            mdn=8, mdo=9, mds=10, mns=1, mne=12, mde=13,
            num_threads=1,
            noe=0, doe=0,  # temporarily disabling output embeddings
        )
    #   When local default fail, try specifying exactly what shared library functions are needed.
    except Exception as exc:
        import logging
        logging.warn(f"Encountered execption with default build of AXY, trying again with an explicit library search.\n  {exc}")
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
    _dependencies = ["real_precision.f90", "blas.f", "lapack.f", "slatec.f", "delsparse.f90"]
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


# Function to run during module setup.
def build_all():
    # Try compiling the 'fmath' library.
    import sys
    _first = sys.path.pop(0)
    try:
        build_unique()
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


# Function for cleaning (before or after a build).
def clean_all():
    import os
    import platform
    import shutil
    import tempfile
    # Get the starting directory.
    start_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(parent_dir)
    # Move files from the provided list of paths to a temporary directory and print the directory path.
    paths = [
        # Unique.
        "tlux/unique/unique.so",
        # AXY.
        "tlux/approximate/axy/axy",
        "tlux/approximate/axy/axy_random",
        "tlux/approximate/axy/axy.mod",
        "tlux/approximate/axy/matrix_operations.mod",
        "tlux/approximate/axy/pcg32_module.mod",
        "tlux/approximate/axy/profiler.mod",
        "tlux/approximate/axy/random.mod",
        "tlux/approximate/axy/sort_and_select.mod",
        "tlux/approximate/axy/test/test_axy_random",
        "tlux/approximate/axy/test/test_axy_module",
        "tlux/approximate/axy/test/test_axy_profiler",
        "tlux/approximate/axy/test/test_axy_random_OLD",
        "tlux/approximate/axy/test/test_axy_module_OLD",
        "tlux/approximate/axy/test/test_axy_profiler_OLD",
        # Ball Tree.
        "tlux/approximate/balltree/fast_select",
        "tlux/approximate/balltree/fast_sort",
        "tlux/approximate/balltree/prune",
        "tlux/approximate/balltree/ball_tree",
        # # Delaunay.
        # f"tlux/approximate/delaunay/delsparse",
        # Regex.
        "tlux/regex/libregex.so",
    ]
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # For each path in the provided list
    for path in paths:
        try:
            # Check if the path exists
            if os.path.exists(path):
                # Maintain the relative path structure within the temporary directory and move the file or directory
                relative_path = os.path.relpath(path)
                destination_path = os.path.join(temp_dir, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                print(f"  (tlux.setup) cleaning '{path}' to '{destination_path}'..", flush=True)
                shutil.move(path, destination_path)
        except Exception as e:
            # Handle any exceptions that may occur during the file/directory moving process
            print(f"ERROR(tlux.setup): Exception while moving '{path}': {str(e)}", flush=True)
    # Move back to the original directory.
    os.chdir(start_dir)


# Function for orchestrating setup.
def setup():
    clean_all()
    build_all()

