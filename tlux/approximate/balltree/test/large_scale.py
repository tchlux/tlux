import sys
import numpy as np
np.set_printoptions(linewidth=sys.maxsize)

from tlux.system import Timer
from tlux.approximate.balltree import BallTree
from tlux.plot import Plot


if __name__ == "__main__":

    np.random.seed(3)

    t = Timer()

    n = 100
    d = 768 // 2
    x = np.random.normal(size=(n,d)).astype("float32")

    tree = BallTree(x)

    t.start()
    distances, indices = tree.nearest(x[:100], k=10, budget=10000)  # Guaranteed to find exact matches.
    print("t.total: ", t.total, flush=True)

    t.start()
    distances, indices = tree.nearest(x[:100], k=10)  # Exhaustive (slower).
    print("t.total: ", t.total, flush=True)

    t.start()
    print(tree.medians[tree.medians > 0].min())
    print()
    print(distances)

    print(tree)

    i = 1
    print(np.linalg.norm(tree.tree[:,indices[i,0]] - tree.tree[:,i]))
    print(np.linalg.norm(tree.tree[:,indices[i,1]] - tree.tree[:,i]))

    exit()



    for i in range(10):
        tree.prune(method="distance", min_distance=26, build=False)
        tree.build(reorder=False)
        print(i, tree)



    p = Plot("Distribution of distances")
    p.add_func("Medians", lambda x: np.percentile(tree.medians, x.flatten()).flatten(), [0, 100], vectorized=True)
    p.add_func("Radii", lambda x: np.percentile(tree.radii, x.flatten()).flatten(), [0, 100], vectorized=True)
    p.show(file_name="/tmp/image_distribution.html")

    print()
    t.start()
    distances, indices = tree.nearest(x[:100], k=10)  # Exhaustive (slower).
    print("t.total: ", t.total, flush=True)
    print()
    print(distances)
