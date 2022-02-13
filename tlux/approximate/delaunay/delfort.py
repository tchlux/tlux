import os
import numpy as np
from tlux.approximate.base import Approximator


# Wrapper class for using the Delaunay fortran code
class Delaunay(Approximator):
    os.environ["OMP_NESTED"] = "TRUE"
    def __init__(self):
        # from util.approximate.delaunay import delsparse
        import fmodpy
        cwd = os.path.dirname(os.path.abspath(__file__))
        self.delsparse = fmodpy.fimport(os.path.join(cwd,"delsparse.f90"), lapack=True, 
                                        blas=True, omp=True, output_dir=cwd, verbose=False,
                                        f_compiler_args="-std=legacy -fPIC -shared -O3",
        )
        # Set up the algorithm for parallel or serial evaluation.
        self.delaunay = self.delsparse.delaunaysparses
        # Initialize containers.
        self.x = None
        self.y = None
        self.errs = {}

    # Use fortran code to compute the boxes for the given data
    def _fit(self, x, y):
        self.x = np.asarray(x, dtype="float64", order="C").T
        self.y = np.asarray(y, dtype="float64", order="C").T
        self.errs = {}

    # Return just the points and the weights associated with them for
    # creating the correct interpolation
    def _predict(self, x, allow_extrapolation=True, print_errors=True,
                 eps=2**(-26), ibudget=10000, extrap=100.0, **kwargs):
        # Get the predictions from Delaunay Fortran code.
        d = self.x.shape[0]
        n = self.x.shape[1]
        pts = np.array(self.x, order="F")
        m = x.shape[0]
        q = np.array(x.T, dtype="float64", order="F")
        simps = np.ones(shape=(d+1, m), dtype="int32", order="F")
        weights = np.ones(shape=(d+1, m), dtype="float64", order="F")
        ierr = np.ones(shape=(m,), dtype="int32", order="F")
        ir = self.y.shape[0]
        interp_in = np.array(self.y, order="F")
        y = np.zeros(shape=(ir,m), dtype="float64", order="F")
        self.delaunay(d, n, pts, m, q, simps, weights, ierr,
                      interp_in=interp_in, interp_out=y,
                      ibudget=ibudget, eps=eps, extrap=extrap,
                      **kwargs)
        # Check for extrapolation.
        assert (allow_extrapolation or ((ierr==1).sum() == 0)), "Encountered extrapolation points when making Delaunay prediction."
        ierr = np.where(ierr == 1, 0, ierr)
        # Handle any errors that may have occurred.
        if (ierr.sum() > 0):
            if print_errors:
                unique_errors = sorted(np.unique(ierr))
                print(" [Delaunay errors:",end="")
                for e in unique_errors:
                    if (e == 0): continue
                    print(" %3i"%e,"at","{"+",".join(tuple(
                        str(i) for i in range(len(ierr))
                        if (ierr[i] == e)))+"}", end=";")
                print("] ")
            # Reset the errors to simplex of 1s (to be 0) and weights of 0s.
            bad_indices = (ierr > 0)
            y[:,bad_indices] = 0.0
        # Return the values.
        return y.T


# Define a simple test to make sure it is working.
if __name__ == "__main__":
    from tlux.plot import Plot
    from tlux.random import well_spaced_box, well_spaced_ball

    # Generate some data.
    x = well_spaced_box(40, 2)
    y = np.cos(10*np.linalg.norm(x, axis=1))

    # Fit a nearest neighbor model.
    m = Delaunay()
    m.fit(x, y)

    # Create a visual.
    p = Plot()
    p.add("data", *x.T, y)
    p.add_func("fit", m, *([[0,1]]*x.shape[1]), vectorized=True, plot_points=5000)
    p.show(z_range=[-2,4])
