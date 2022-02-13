import numpy as np
from tlux.approximate.base import Approximator

# Wrapper class for using the Delaunay scipy code.
class Delaunay(Approximator):
    def __init__(self):
        from scipy.spatial import Delaunay as qHullDelaunay
        self.qHullDelaunay = qHullDelaunay
        self.x = None
        self.y = None

    # Use fortran code to compute the boxes for the given data
    def _fit(self, x, y):
        # Delaunay is only made for more than 1 dimension.
        if (x.shape[1] > 1):
            self.mesh = self.qHullDelaunay(x)
        else:
            # Define a 1D function that linearly interpolates the data.
            x = x.flatten()
            i = np.argsort(x)
            x = x[i]
            y = y[i]
            gaps = x[1:] - x[:-1]
            def interpolate(z, x=x, y=y):
                v = []
                for p in z.flatten():
                    if   (p <= x[0]):  v.append(y[0])
                    elif (p >= x[-1]): v.append(y[-1])
                    else:
                        i = np.searchsorted(x, p, side="right")
                        rval = (p - x[i-1])
                        lval = (x[i] - p)
                        v.append( (lval*y[i-1] + rval*y[i]) / (lval+rval) )
                v = np.asarray(v)
                return v
            # Store this 1D function.
            self.mesh = interpolate
        # Store the x and y values for prediction.
        self.x = x
        self.y = y
        self.y_min = y.min(axis=0)
        self.y_max = y.max(axis=0)


    # Function that returns the indices of points and the weights that
    # should be used to make associated predictions for each point in
    # "points".
    def _predict(self, x):
        if (len(self.x.shape) > 1):
            # Solve for the weights in the Delaunay model
            simp_ind = self.mesh.find_simplex(x)
            # If a point is outside the convex hull, use the
            # closest simplex to extrapolate the value
            simp_ind = np.where(
                simp_ind == -1,
                np.argmax(self.mesh.plane_distance(x), axis=1),
                simp_ind
            )
            simp = self.mesh.simplices[simp_ind]
            zx = self.x[simp.flatten()].reshape(simp.shape+(self.x.shape[-1],))
            # Solve for the response value with a linear solve.
            system = np.concatenate((zx,np.ones(zx.shape[:-1]+(1,))), axis=-1)
            system = np.transpose(system, axes=(0,2,1))
            x_pts = np.concatenate((x,np.ones(x.shape[:-1]+(1,))), axis=-1)
            weights = np.linalg.solve(system, x_pts)
            # Get the y values.
            zy = self.y[simp.flatten()].reshape(simp.shape+(self.y.shape[-1],))
            y = np.zeros((x.shape[0], zy.shape[-1]))
            for i in range(x.shape[0]):
                y[i,:] = np.matmul(zy[i].T, weights[i,:])
            # Clip to not leave range of observed values.
            return np.clip(y, self.y_min, self.y_max)
        else:
            return self.mesh(x)


# Define a simple test to make sure it is working.
if __name__ == "__main__":
    from tlux.plot import Plot
    from tlux.random import well_spaced_box, well_spaced_ball

    # Generate some data.
    x = well_spaced_box(400, 2)
    y = np.cos(10*np.linalg.norm(x, axis=1))

    # Fit a nearest neighbor model.
    m = Delaunay()
    m.fit(x, y)

    # Create a visual.
    p = Plot()
    p.add("data", *x.T, y)
    p.add_func("fit", m, *([[0,1]]*x.shape[1]), vectorized=True)
    p.show(z_range=[-2,4])
