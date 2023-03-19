import os, sys, time
import numpy as np
import fmodpy
from tlux.random import well_spaced_ball as random
from tlux.random import ball as random
from memory_profiler import profile

# Get the matrix operations code from the main directory.
source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
source_path = os.path.join(source_dir, "matrix_operations.f90")
mops = fmodpy.fimport(source_path, blas=True, lapack=True, omp=True).matrix_operations
# Load an old copy of matrix operations that has been used heavily.
old_mops = fmodpy.fimport("old_matrix_operations.f90", blas=True, lapack=True, omp=True).matrix_operations

print("Initializing data..")
n = 10000000
d = 64
x = np.asarray(random(n, d).T, dtype="float32", order="F")
shift = np.zeros((d,), dtype="float32")
vecs = np.zeros((d,d), dtype="float32", order="F")

print("Calling radialize..")
def run():
    start = time.time()    
    _new = mops.radialize(x=np.array(x, order="F"), shift=shift.copy(), vecs=np.array(vecs, order="F"), max_to_square=n)
    total = time.time() - start
    start = time.time()    
    _old = old_mops.radialize(x=np.array(x, order="F"), shift=shift.copy(), vecs=np.array(vecs, order="F"))
    old_total = time.time() - start
    for a,b in zip(_old, _new):
        diff = np.abs(a-b).max()
        assert diff < 2**(-14), f"Difference between implementations ({diff:.8f}) is too large."
    print(f"Comparison passed, no issues.. ({old_total/total:.1f}x speedup)")
    return total
print(f"Done in {run():.2f} seconds.")
