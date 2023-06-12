import numpy as np
import fmodpy

# Matrix operations module.
random = fmodpy.fimport("../axy_random.f90", dependencies=["pcg32.f90"], name="test_axy_random").random


# --------------------------------------------------------------------
#                            RANDOM_INTEGER

# Make sure that the random integer generation has desired behavior.
def _test_random_integer():
    print("RANDOM_INTEGER..", flush=True)
    counts = {}
    # trials = 100
    trials = 1000000
    bins = 3
    for i in range(trials):
        val = random.random_integer(max_value=bins)
        counts[val] = counts.get(val,0) + 1
    total_count = sum(counts.values())
    for (v,c) in sorted(counts.items()):
        ratio = (c / total_count)
        error = abs(ratio - 1/bins)
        assert (error < 0.001), f"Bad ratio of random integers had value {v} when generating with max_value = {bins}.\n  Ratio was {ratio}\n  Expected {1/bins}"
    print("  passed.", flush=True)


# --------------------------------------------------------------------
#                            RANDOM_REAL

# Make sure that random real number generation has the desired behavior.
def _test_random_real():
    print("RANDOM_REAL..", flush=True)    
    # trials = 10
    trials = 100000
    # 
    # Generate random numbers.
    random.seed_random()
    vals = []
    for i in range(trials):
        vals.append( random.random_real(v=True)[1] )
    # Sort all the random numbers generated.
    vals = sorted(vals)
    # Get the list of linearly spaced real numbers of the same sample size over the unit interval.
    expected = [(i+1)/trials - (1/(2*trials))  for i in range(trials)]
    # Compute the absolute differences between the the sorted values and the evenly spaced ones.
    diffs = [abs(v-e) for (v,e) in zip(vals, expected)]
    # Assert that the maximum difference between the values and a uniform grid is small.
    assert (max(diffs) < 0.01), f"Unexpectedly lumpy distribution of random real numbers for scalar output."
    # 
    # Now repeat, generating all random numbers at once.
    random.seed_random(30183010)
    r = np.zeros((max(1,trials // (20*30)), 20, 30), dtype="float32", order="F")
    random.random_real(r=r, s=r.size)
    vals = sorted(r.flatten())
    expected = [(i+1)/r.size - (1/(2*r.size))  for i in range(r.size)]
    diffs = [abs(v-e) for (v,e) in zip(vals, expected)]
    assert (max(diffs) < 0.01), f"Unexpectedly lumpy distribution of random real numbers for tensor output."
    print("  passed.", flush=True)


# --------------------------------------------------------------------
#                    INDEX_TO_PAIR     PAIR_TO_INDEX

def _test_index_to_pair():
    print("INDEX_TO_PAIR")
    mv = 30
    all_pairs = set()
    # Verify that the pair mapping works forwards and backwards.
    for i in range(1, mv**2+1):
        pair1, pair2 = random.index_to_pair(max_value=mv, i=i)
        all_pairs.add((pair1, pair2))
        j = random.pair_to_index(max_value=mv, pair1=pair1, pair2=pair2)
        assert (i == j), f"Index to pair mapping failed for i={i} max_value={limit} pair={pair} ii={j}."
    # Verify that all pairs were actually generated.
    for i in range(1,mv+1):
        for j in range(1,mv+1):
            assert ((i,j) in all_pairs), f"Pair {(i,j)} missing from enumerated set." 
    print(" passed")


# --------------------------------------------------------------------
#                        INITIALIZE_ITERATOR

def _test_initialize_iterator():
    print("INITIALIZE_ITERATOR..", flush=True)
    nm = 10000
    max_size = 23
    seed = 0
    pairwise = False
    random.seed_random(seed)
    # Initialize space to hold iterators.
    agg_iterators = np.zeros((6,nm), dtype="int64", order="F")    
    # Create sizes.
    sizes = np.asarray(
        [random.random_integer(max_value=max_size) for i in range(nm)],
        dtype="int64", order="F"
    )
    # Initialize all iterators.
    for i in range(nm):
        if (sizes[i] == 0):
            agg_iterators[:,i] = 0
        else:
            agg_iterators[0,i] = sizes[i]
            if pairwise:
                agg_iterators[0,i] = agg_iterators[0,i]**2
            agg_iterators[1:,i] = random.initialize_iterator(
                i_limit=agg_iterators[0,i], seed=seed
            )
    # Check that each aggregator works.
    for i in range(nm):
        if (sizes[i] == 0): continue
        # Generate a full loop of values.
        seen = []
        *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i])
        for _ in range(agg_iterators[0,i]):
            seen.append(next_i)
            *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i], reshuffle=True)
        seen_sorted = tuple( sorted(seen) )
        expected = tuple( range(1,sizes[i]**(2 if pairwise else 1) + 1) )
        assert (seen_sorted == expected), \
            f"Aggregate iterator did not produce expected list of elements.\n  {seen_sorted}\n  {expected}\n  {agg_iterators[:,i]}"
        # Generate another full loop (acknowledging that a reshuffle should have happened).
        past_seen = seen
        seen = []
        *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i])
        for _ in range(agg_iterators[0,i]):
            seen.append(next_i)
            *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i], reshuffle=True)
        seen_sorted = tuple( sorted(seen) )
        assert (seen_sorted == expected), \
            f"Aggregate iterator did not produce expected list of elements.\n  {seen_sorted}\n  {expected}\n  {agg_iterators[:,i]}"
        # Make sure the two loops aren't equal.
        if (sizes[i] > 1):
            assert (tuple(past_seen) != tuple(seen)), f"The two sequences generated were the same.\n  first:  {past_seen}\n  second: {seen}"

    # Done testing.
    print("  passed.", flush=True)


# --------------------------------------------------------------------
#                        RANDOM_UNIT_VECTORS

# TODO: Add test with large number of vectors, ensure no Inf or Nan generated.
def _test_random_unit_vectors():
    print("RANDOM_UNIT_VECTORS..", flush=True)
    # Generate test data.
    n = 5000 # number of points
    for d in (2,): # 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 60, 120):
        bd = 2 # full number of outputs
        br = 1 # reduced rank approximation

        # Random unit vectors.
        a = np.zeros((n,d), dtype="float32")
        random.random_unit_vectors(a.T)

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



if __name__ == "__main__":
    print()
    # expected = [121, 72, 47, 110, 69, 116, 59, 90, 17, 32, 71, 70, 93, 76,
    #             83, 50, 41, 120, 95, 30, 117, 36, 107, 10, 65, 80, 119, 118, 13, 3,
    #             98, 89, 40, 15, 78, 37, 84, 27, 58, 113, 39, 38, 61, 44, 51, 18, 9,
    #             88, 63, 85, 4, 75, 106, 33, 48, 87, 86, 109, 92, 99, 66, 57, 8, 111,
    #             46, 5, 52, 26, 81, 96, 7, 6, 29, 12, 19, 114, 105, 56, 31, 94, 53,
    #             100, 43, 74, 1, 16, 55, 54, 77, 60, 67, 34, 25, 104, 79, 14, 101, 20,
    #             91, 49, 64, 103, 102, 108, 115, 82, 73, 24, 62, 21, 68, 11, 42, 97,
    #             112, 23, 22, 45, 28, 35, 2]
    # # Manually check the expected trajectory of an iterator.
    # iterator = [121, 120, 105, 15, 128, 0]
    # # Generate a full loop of values.
    # seen = set()
    # *iterator[1:], next_i = random.get_next_index(*iterator[:], reshuffle=True)
    # for _ in range(iterator[0]+1):
    #     print(next_i, end=", ", flush=True)
    #     seen.add(next_i)
    #     *iterator[1:], next_i = random.get_next_index(*iterator[:], reshuffle=True)
    # seen = tuple( sorted(seen) )
    # expected = tuple( range(1,iterator[0] + 1) )
    # assert (seen == expected), \
    #     f"Aggregate iterator did not produce expected list of elements.\n  {seen}\n  {expected}\n  {agg_iterators[:,i]}"
    # print()
    # exit()

    _test_random_integer()
    _test_random_real()
    _test_index_to_pair()
    _test_initialize_iterator()
    # _test_random_unit_vectors()
