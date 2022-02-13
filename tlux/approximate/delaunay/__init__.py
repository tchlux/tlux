try:
    from .delfort import Delaunay
    from .qhull import Delaunay as QHullDelaunay
except ImportError:
    from tlux.approximate.delaunay.delfort import Delaunay
    from tlux.approximate.delaunay.qhull import Delaunay as QHullDelaunay


# Define a simple test to make sure it is working.
if __name__ == "__main__":
    import numpy as np
    from tlux.plot import Plot
    from tlux.random import well_spaced_box, well_spaced_ball

    # Generate some data.
    x = well_spaced_box(40, 2)
    y = np.cos(10*np.linalg.norm(x, axis=1))

    # Fit a nearest neighbor model.
    m = Delaunay()
    # m = QHullDelaunay()
    m.fit(x, y)

    # Create a visual.
    p = Plot()
    p.add("data", *x.T, y)
    p.add_func("fit", m, *([[0,1]]*x.shape[1]), vectorized=True, plot_points=5000)
    p.show(z_range=[-2,4])
