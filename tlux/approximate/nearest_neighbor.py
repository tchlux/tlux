import numpy as np
from tlux.approximate.base import Approximator

# Class for computing an interpolation between the nearest n neighbors
class NearestNeighbor(Approximator):
    def __init__(self, k=1):
        self.k = k
        self.points = None
        self.values = None

    # Use sklearn to 
    def _fit(self, x, y, k=None, **kwargs):
        # from balltree import BallTree
        from sklearn.neighbors import BallTree
        if (k is not None): self.k = k
        # Process and store local information
        self.points = x
        self.tree = BallTree(self.points)
        self.values = y

    # Function that returns the indices of points and the weights that
    # should be used to make associated predictions for each point in
    # "points".
    def _predict(self, x):
        indices = self.tree.query(x, k=self.k, return_distance=False).flatten()
        y = self.values[indices]
        return y


# Define a simple test to make sure it is working.
if __name__ == "__main__":
    from tlux.plot import Plot
    from tlux.random import well_spaced_box

    # Generate some data.
    x = well_spaced_box(200, 2)
    y = np.cos(10*np.linalg.norm(x, axis=1))

    # Fit a nearest neighbor model.
    m = NearestNeighbor(k=1)
    m.fit(x, y)

    # Create a visual.
    p = Plot()
    p.add("data", *x.T, y)
    p.add_func("fit", m, [0,1], [0,1])
    p.show()
