import os

# ----------------------------------------------------------------
#  Enable debugging option "-fcheck=bounds".
import fmodpy
# fmodpy.configure(verbose=True)
fmodpy.config.f_compiler_args = "-fPIC -shared -O3 -pedantic -fcheck=bounds -ftrapv -ffpe-trap=invalid,overflow,underflow,zero"
# fmodpy.config.link_blas = "-framework Accelerate"
# fmodpy.config.link_lapack = "-framework Accelerate"
_dependencies = ["random.f90", "matrix_operations.f90", "sort_and_select.f90", "axy.f90"]
_dir = os.path.dirname(os.path.realpath(__file__))
_path = os.path.join(_dir, "axy.f90")
_axy = fmodpy.fimport(
    _path, dependencies=_dependencies, output_dir=_dir,
    blas=True, lapack=True, omp=True, wrap=True,
    rebuild=False,
    # verbose=True, 
    # link_blas="", link_lapack="",
    # libraries = [_dir] + fmodpy.config.libraries,
    # symbols = [
    #     ("sgemm", "blas"),
    #     ("sgels", "lapack"),
    #     ("omp_get_max_threads", "omp")
    # ],
# )
# ----------------------------------------------------------------
