import numpy as np
import fmodpy

# Matrix operations module.
rand = fmodpy.fimport("../axy_random.f90").random
print()


# Make sure that the random integer generation has desired behavior.
def _test_random_integer():
    print("Random integer..", flush=True)
    counts = {}
    trials = 1000000
    bins = 3
    for i in range(trials):
        val = rand.random_integer(max_value=bins)
        counts[val] = counts.get(val,0) + 1
    total_count = sum(counts.values())
    for (v,c) in sorted(counts.items()):
        ratio = (c / total_count)
        error = abs(ratio - 1/bins)
        assert (error < 0.001), f"Bad ratio of random integers had value {v} when generating with max_value = {bins}.\n  Ratio was {ratio}\n  Expected {1/bins}"
    print("  passed.", flush=True)

_test_random_integer()


# --------------------------------------------------------------------
#                    INDEX_TO_PAIR     PAIR_TO_INDEX

def _test_index_to_pair():
    print("INDEX_TO_PAIR")
    mv = 30
    all_pairs = set()
    # Verify that the pair mapping works forwards and backwards.
    for i in range(1, mv**2+1):
        pair1, pair2 = rand.index_to_pair(max_value=mv, i=i)
        all_pairs.add((pair1, pair2))
        j = rand.pair_to_index(max_value=mv, pair1=pair1, pair2=pair2)
        assert (i == j), f"Index to pair mapping failed for i={i} max_value={limit} pair={pair} ii={j}."
    # Verify that all pairs were actually generated.
    for i in range(1,mv+1):
        for j in range(1,mv+1):
            assert ((i,j) in all_pairs), f"Pair {(i,j)} missing from enumerated set." 
    print(" passed")

_test_index_to_pair()



# --------------------------------------------------------------------
#                        RANDOM_UNIT_VECTORS

# TODO: Add test with large number of vectors, ensure no Inf or Nan generated.
def _test_random_unit_vectors():
    print("Random unit vectors..", flush=True)
    # Generate test data.
    n = 5000 # number of points
    for d in (2,): # 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 60, 120):
        bd = 2 # full number of outputs
        br = 1 # reduced rank approximation

        # Random unit vectors.
        a = np.zeros((n,d), dtype="float32")
        rand.random_unit_vectors(a.T)

        # Overwrite a with perfectly spaced version for 2 dimensions.
        if (d == 2):
            _ = np.linspace(0, 2*np.pi, n)
            a[:,0] = np.cos(_)
            a[:,1] = np.sin(_)

        # Assert that the distance from the origin for each point is one.
        lengths = np.linalg.norm(a, axis=1)
        length_diff = abs(lengths - 1.0)
        index = np.argmax(length_diff)
        tolerance = 0.0001
        assert (max(length_diff) < tolerance), \
            "One of the vector lengths was significantly different than 1.\n" \
            f"  index = {index}\n" \
            f"  length = {lengths[index]}\n" \
            f"  tolerance = {tolerance}"

        # Assert that the distribution of distances between pairs is not
        #  significantly different than the expected analytic solution for
        #  distances between all pairs of points on a sphere.
        distances = np.concatenate([
            np.linalg.norm(a[:i,:] - a[i], axis=1)
            for i in range(1,len(a))
        ] + [
            np.linalg.norm(a[i+1:,:] - a[i], axis=1)
            for i in range(len(a)-1)
        ], axis=0)


        # If either fail, then plot the vectors.
        from tlux.math import project
        from tlux.plot import Plot, multiplot
        # Plot the actual points (in 3d).
        title = f"3D projection of {d}D points    |    CDF of distances between pairs of {d}D points"
        p1 = Plot(title)
        vecs = project(a, 3)
        if (n <= 100):
            for i in range(n):
                l = np.vstack([[0]*3, vecs[i]])
                p1.add(str(i), *l.T, mode="lines", show_in_legend=False)
        else:
            p1.add("vecs", *vecs.T, marker_size=2, color=1)
        # Plot the distribution of distances.
        p2 = Plot(title, "Distance between pair of points", "P[||p1-p2||] <= Distance")

        # Create the function that converts x and y positions into distances.
        radians = np.linspace(np.pi, 0, 1000)
        x_values = 1 + np.cos(radians)  # [0, ..., 2]
        y_values = np.sin(radians)  # [0, ..., 1, ..., 0]
        dist_values = np.linalg.norm(np.vstack((x_values, y_values)), axis=0)  # [0, ..., 2]

        # def cdf(dist):
        #     # Infer the "x" by pushing the value from the inverse function for distances.
        #     x = np.interp(dist, xp=dist_values, fp=x_values)
        #     # Map the x ∈ [0, 2] to the density.
        #     if (x <= 1): # x ** (1/d) 
        #         return ((1 - (1-x)**(1/(d-1))) / 2)
        #     else:
        #         return ((1/2) + ((x-1) ** (1/(d-1))) / 2)
        # p2.add_func("truth_guess", cdf, [0, 2], color=0, dash="dot")

        p2.add("coordinates", x_values, y_values, mode="lines", color=2)
        p2.add("dist(x)", x_values, dist_values, mode="lines", color=3)
        p2.add("xdist(d)", dist_values, x_values, mode="lines", color=4)

        # Add the distribution of distances.
        cdf_y = np.linspace(0, 1, 1000)
        cdf_x = np.percentile(distances, 100 * cdf_y)
        p2.add("distances CDF", cdf_x, cdf_y, mode="lines", color=1, show_in_legend=False)
        # Show both plots.
        multiplot([p1, p2], show_legend=False)

_test_random_unit_vectors()

