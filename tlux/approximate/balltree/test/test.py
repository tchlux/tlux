import numpy as np


# Function for verifying correctness.
def small_diff(v1,v2): return abs(v1 - v2) < .01


def test_tree(LARGE_TEST=False):
    from tlux.approximate.balltree import BallTree 
    np.random.seed(0)

    # if LARGE_TEST: size = (20000000,10)
    # if LARGE_TEST: size = (100000,1000)
    if LARGE_TEST: size = (50000000, 32)
    else:          size = (4, 2)
    print()
    print(f"Allocating array.. {size}", flush=True)

    x = np.random.random(size=size).astype("float32")

    # Handle printout based on test size.
    if len(x) < 20:
        print()
        print("x:",x.T.shape,"\n",x)
        print()

    print("Building tree..", flush=True)
    from tlux.system import Timer
    t = Timer()
    t.start()

    k = 3
    leaf_size = 1
    tree = BallTree(x, leaf_size=leaf_size, reorder=True)

    t.stop()
    print(f"done in {t()} seconds.", flush=True)

    if len(x) < 20:
        print()
        print('-'*70)
        print("Tree:")
        print(tree.tree.T)
        print()
        print("Usage:")
        print(tree.usage)
        print()
        print("Order, Radius:")
        print(tree.order)
        print(tree.radii)
        print('-'*70)

    z = np.random.random(size=(3,size[1]))

    if not LARGE_TEST: print("\nz: ",z,"\n")
    
    t.start()
    d,i = tree.query(z, k=k)
    t.stop()
    print("ball tree query in",t(),"seconds")
    if len(x) < 20: print(" usage:", tree.usage)
    d,i = d[0], i[0]
    # Do an approximate query.
    t.start()
    _,_ = tree.query(z, k=k, budget=100)
    t.stop()
    print("ball tree constrained query in",t(),"seconds")
    if len(x) < 20: print(" usage:", tree.usage)
    # Measure the "true" distances between points.
    print("linear search in ...", end="\r", flush=True)
    t.start()
    true_dists = np.linalg.norm(x - z[0,:], axis=1)
    t.stop()
    print("linear search in",t(),"seconds")
    print()
    print("Tree/Truth")
    print("i: \n",i,"\n",np.argsort(true_dists)[:k])
    assert all(i == np.argsort(true_dists)[:k])
    print("d: \n",d,"\n",np.sort(true_dists)[:k])
    assert np.allclose(d, np.sort(true_dists)[:k])


def test_compare_against_sklearn(LARGE_TEST=False):
    print()
    print("="*70)

    from tlux.system import Timer
    t = Timer()

    if LARGE_TEST: train, dim = 1000000, 512
    else:          train, dim = 5, 2
    test = 10
    leaf_size = 1
    k = 5
    print("Initializing data..", flush=True)
    np.random.seed(0)
    x = np.random.random(size=(train,dim)).astype("float32")
    z = np.random.random(size=(test,dim)).astype("float32")  # Use random z.
    # z = x[np.random.randint(0,len(x),size=(test,))]  # Use z points from x.
    print()
    print("x:", x.shape)
    print("z:", z.shape)
    # ----------------------------------------------------------------
    from tlux.approximate.balltree import BallTree as BT
    print()
    print("Fortran Ball Tree")
    t.start()
    # tree = BT(x, leaf_size=leaf_size)
    tree = BT(x[:len(x)//2], leaf_size=leaf_size, build=True) # build=False
    tree.add(x[len(x)//2 : (3*len(x))//4])
    tree.add(x[(3*len(x))//4 :])
    # tree.build()
    ct = t.stop()
    print("Construction time:", ct)
    t.start()
    d, i = tree.query(z, k=k)
    qt = t.stop()
    print("Query time:       ", qt)
    print("d: ",d[0])
    print("i: ",i[0])
    d1, i1 = d[0].copy(), i[0].copy()
    # ----------------------------------------------------------------
    from sklearn.neighbors import BallTree
    print()
    print("Sklearn Ball Tree")
    t.start()
    tree = BallTree(x, leaf_size=leaf_size)
    ct = t.stop()
    print("Construction time:", ct)
    t.start()
    d, i = tree.query(z, k=k)
    qt = t.stop()
    print("Query time:       ", qt)
    print("d: ",d[0])
    print("i: ",i[0])
    d2, i2 = d[0].copy(), i[0].copy()
    # ----------------------------------------------------------------
    print()
    print("Brute Force")
    # Convert to float64 (for regular test this does nothing, for integer...
    t.start()
    d = np.sqrt(np.sum(x**2, axis=1, keepdims=True) + np.sum(z**2, axis=1, keepdims=True).T - 2 * np.dot(x, z.T))[:,0]
    i = np.argsort(d)
    qt = t.stop()
    print("Query time:", qt)
    i = i[:k]
    d = d[i]
    print("d: ",d)
    print("i: ",i)
    d3, i3 = d.copy(), i.copy()
    # ----------------------------------------------------------------
    max_diff = max(max(abs(d1-d2)), max(abs(d1-d3)))
    ds_match = max_diff < 2**(-13) # 2**(-26)
    is_match = np.all(i1 == i2) and np.all(i1 == i3)
    print()
    print(f"Max difference in distance calculations:\n   {max_diff:.3e}")
    assert (ds_match and is_match), f"\nERROR\n  is_match: {is_match}\n  ds_match: {ds_match} {max(abs(d1-d3)):.3e} {max(abs(d1-d2)):.3e}"


def test_sort(LARGE_TEST=False):
    if LARGE_TEST: N = 1000000
    else:          N = 10
    from tlux.approximate.balltree import fast_sort
    # Import a timer.
    from tlux.system import Timer
    t = Timer()
    # Generate test numbers.
    print()
    print(f"Generating {N} numbers..", flush=True)
    x = np.random.random(size=N).astype("float32")
    i = np.arange(len(x)) + 1
    print()
    # Test the fortran code.
    pts = x.copy()
    ids = i.copy()
    t.start()
    pts, ids = fast_sort.argsort(pts, ids)
    t.stop()
    # Check for correctness.
    ids_match = np.all(x[ids-1] == pts)
    is_sorted = np.all(np.diff(pts)>=0)
    try: assert(ids_match and is_sorted)
    except:
        print("ERROR")
        print(" ids_match: ",ids_match)
        print(" is_sorted: ",is_sorted)
        print()
        print("x:  ", x[ids-1])
        print("pts:", pts)
        print("ids:", ids)
        print()
        print(x)
        print(x[ids-1])
        exit()
    print("argsort: %.6f"%(t.total))
    # Test the NumPy code.
    pts = x.copy()
    ids = i.copy()
    t.start()
    ids = pts.argsort()
    t.stop()
    print("numpy:   %.6f"%(t.total))


def test_prune_level():
    from tlux.approximate.balltree import BallTree, prune
    
    tree_size = 14
    level = 3
    indices = np.zeros(2**level, order='F', dtype=np.int64)
    indices, found = prune.level(tree_size, level, indices, found=0)
    assert(found == 7)
    assert(tuple(indices) == (4,5,7,8,11,12,14,0))


def test_prune_inner():
    from tlux.approximate.balltree import BallTree, prune
    
    tree_size = 14
    level = 3
    indices = np.zeros(2**level, order='F', dtype=np.int64)
    indices, found = prune.level(tree_size, level, indices, found=0)

    # Build a tree over random points in 1 dimension.
    np.random.seed(3)
    pts = np.random.random(size=(100000,1))
    tree = BallTree()
    tree.add(pts)
    tree.build()

    vals = sorted((tree[0], tree[2], tree[1],
                  tree[(tree.tree.shape[1] + 2) // 2 + 1],
                  tree[(tree.tree.shape[1] + 2) // 2]
    ))
    print()
    for v in vals: print(v)
    print()
    assert(all(small_diff(v1,v2) for (v1,v2) in zip(
        vals, np.linspace(0,1,5))))

    tree.prune(4, method="inner")
    print(tree)
    print(tree[:len(tree)][0,:])
    print(sorted(tree[:len(tree)][0,:]))
    print()
    assert(all(small_diff(v1,v2) for (v1,v2) in zip(
        sorted(tree[:len(tree)][0,:]), np.linspace(0,1,len(tree)))))

    tree.prune(3, method="inner")
    print(tree)
    print(tree[:len(tree)][0,:])
    print(sorted(tree[:len(tree)][0,:]))
    print()
    assert(all(small_diff(v1,v2) for (v1,v2) in zip(
        sorted(tree[:len(tree)][0,:]), np.linspace(0,1,len(tree)))))

    tree.prune(2, method="inner")
    print(tree)
    print(tree[:len(tree)][0,:])
    print(sorted(tree[:len(tree)][0,:]))
    print()
    assert(all(small_diff(v1,v2) for (v1,v2) in zip(
        sorted(tree[:len(tree)][0,:]), np.linspace(0,1,len(tree)))))

    tree.prune(1, method="inner")
    print(tree)
    print(tree[:len(tree)][0,:])
    print(sorted(tree[:len(tree)][0,:]))
    print()
    assert(small_diff(tree[0][0], .5))


def test_prune_distance():
    import math
    from tlux.approximate.balltree import BallTree, prune
    from tlux.plot import Plot
    

    # Build a tree.
    # 
    #  Get the distances to furthest inner and furthest outer child for each point.
    #   [optional] Visualize them in 2D to verify.
    # 
    #  Confirm a furthest outer is pruned correctly.
    # 
    #  Confirm a furthest inner is pruned correctly.
    # 
    #  Confirm a node with children is pruned correctly (keep parent, no more children).
    # 

    # Plot the tree and all levels.
    def show_tree(tree, show=True):
        tree_size = tree.size
        p = Plot()
        for max_level in range(0, math.ceil(math.log(tree_size, 2))):
            for current_level in range(0, math.ceil(math.log(tree_size, 2))):
                # Get the new point indices (ensuring they haven't already been shown).
                indices = np.zeros(2**current_level, order='F', dtype=np.int64)
                indices, found = prune.level(tree_size, current_level, indices, found=0)
                if (found > 0):
                    # Convert to python indices (from Fortran).
                    indices = indices[:found] - 1 
                    # Add the region for the balls at this level.
                    if (current_level == max_level):
                        for i in indices:
                            p.add_region(
                                f"ball {i}",
                                lambda x: np.linalg.norm(x-tree.tree[:,i]) <= tree.medians[i],
                                min_max_xy,
                                min_max_xy,
                                # line_width=1,
                                plot_points=10000,
                                nonconvex=True,
                                color=1,
                                group=current_level, # f"ball {i}",
                                show_in_legend=False,
                                frame=max_level,
                            )
                    # Add the points.
                    pts = tree.tree[:,indices]
                    if pts.size > 0:
                        p.add(f"level {current_level}", *pts,
                              color=(0 if current_level == max_level else 1),
                              marker_size=4, marker_line_width=1,
                              group=current_level, frame=max_level)
        if len(p.data) > 0:
            p.show(aspect_mode="cube", append=True, show=show)

    np.random.seed(1)
    tree_size = 213
    points = np.random.normal(size=(tree_size, 2)).astype("float32")
    tree = BallTree(points, selector=BallTree.SELECT_NEAREST_POINT)
    min_max_xy = [-3, 3]

    def min_dists(tree):
        from tlux.math import pairwise_distance
        pts = tree.tree[:,tree.order[:tree.size]-1].T
        distances = pairwise_distance(pts)
        return sorted(set(np.min(distances, axis=1)))

    print()
    print("tree.size:    ", tree.size, flush=True)
    print("tree.order:   ", tree.order[:tree.size]-1, flush=True)
    print("tree.medians: ", tree.medians[tree.order[:tree.size]-1], flush=True)
    print("tree.radii:   ", tree.radii[tree.order[:tree.size]-1], flush=True)
    print(min_dists(tree))
    show_tree(tree)

    even_selector = BallTree.SELECT_RANDOM_POINT
    odd_selector = BallTree.SELECT_FURTHEST_POINT

    limit = 10
    for i in range(limit):
        if ((i % 2) == 0):
            selector = even_selector
        else:
            selector = odd_selector
        tree.prune(method="distance", min_distance=0.1, build=False)
        tree.build(selector=selector, reorder=True)
        print()
        print("selector: ", selector, flush=True)
        print("tree.size:    ", tree.size, flush=True)
        print("tree.order:   ", tree.order[:tree.size]-1, flush=True)
        print("tree.medians: ", tree.medians[tree.order[:tree.size]-1], flush=True)
        print("tree.radii:   ", tree.radii[tree.order[:tree.size]-1], flush=True)
        print(min_dists(tree))
        show_tree(tree, show=(i == limit-1))


# × SUM(SQ) - 3.17 2.37 2.36 2.33 2.33
# ✓ OMP     - 2.51 2.31 2.30 2.26 2.27
# × PRE-ADD - 2.35 2.33 2.31 2.27 2.35
# 
if __name__ == "__main__":
    LARGE_TEST = True

    # test_tree(LARGE_TEST=LARGE_TEST)
    test_compare_against_sklearn(LARGE_TEST=LARGE_TEST)
    # test_sort(LARGE_TEST=LARGE_TEST)
    # test_prune_level()
    # test_prune_inner()
    # test_prune_distance()
