# Ball Tree wrapper for Fortran module.

# TODO: Investigate the correctness of the partial tree traversal
#       (budget < size(order)) when there are duplicate data points.
# TODO: Make queries over trees with size > built do a brute-force
#       search over the unbuilt portion of the tree (when exact).

import os
import numpy as np

# Count the number of CPUs for setting default parallelism.
from multiprocessing import cpu_count

# Import the Fortran utilities.
try:
    from tlux.approximate.balltree.ball_tree import ball_tree
    from tlux.approximate.balltree.fast_sort import fast_sort
    from tlux.approximate.balltree.fast_select import fast_select
    from tlux.approximate.balltree.prune import prune
except:
    from tlux.setup import build_balltree
    ball_tree, fast_sort, fast_select, prune = build_balltree()

# ------------------------------------------------------------------
#                        FastSort method
# 
# This routine uses a combination of QuickSort (with modestly
# intelligent pivot selection) and Insertion Sort (for small arrays)
# to achieve very fast average case sort times for both random and
# partially sorted data. The pivot is selected for QuickSort as the
# median of the first, middle, and last values in the array.
# 
# Arguments:
# 
#   VALUES   --  A 1D array of real numbers.
# 
# Optional:
# 
#   INDICES  --  A 1D array of original indices for elements of VALUES.
#   MIN_SIZE --  An positive integer that represents the largest
#                sized VALUES for which a partition about a pivot
#                is used to reduce the size of a an unsorted array.
#                Any size less than this will result in the use of
#                INSERTION_ARGSORT instead of ARGPARTITION.
# 
# Output:
# 
#   The elements of the array VALUES are sorted and all elements of
#   INDICES are sorted symmetrically (given INDICES = 1, ...,
#   SIZE(VALUES) beforehand, final INDICES will show original index
#   of each element of VALUES before the sort operation).
# 
def argsort(values, indices=None, min_size=None):
    indices = np.arange(len(values))
    return fast_sort.argsort(values, indices, min_size=min_size)


# ------------------------------------------------------------------
#                       FastSelect method
# 
# Given VALUES list of numbers, rearrange the elements of VALUES
# such that the element at index K has rank K (holds its same
# location as if all of VALUES were sorted). Symmetrically rearrange
# array INDICES to keep track of prior indices.
# 
# This algorithm uses the same conceptual approach as Floyd-Rivest,
# but instead of standard-deviation based selection of bounds for
# recursion, a rank-based method is used to pick the subset of
# values that is searched. This simplifies the code and improves
# readability, while achieving the same tunable performance.
# 
# Arguments:
# 
#   VALUES   --  A 1D array of real numbers.
#   K        --  A positive integer for the rank index about which
#                VALUES should be rearranged.
# Optional:
# 
#   INDICES  --  A 1D array of original indices for elements of VALUES.
#   DIVISOR  --  A positive integer >= 2 that represents the
#                division factor used for large VALUES arrays.
#   MAX_SIZE --  An integer >= DIVISOR that represents the largest
#                sized VALUES for which the worst-case pivot value
#                selection is tolerable. A worst-case pivot causes
#                O( SIZE(VALUES)^2 ) runtime. This value should be
#                determined heuristically based on compute hardware.
# 
# Output:
# 
#   The elements of the array VALUES are rearranged such that the
#   element at position VALUES(K) is in the same location it would
#   be if all of VALUES were in sorted order. Also known as,
#   VALUES(K) has rank K.
# 
def argselect(values, k, indices=None, divisor=None, max_size=None):
    if (indices is None): indices = np.arange(len(values))
    return fast_select.argselect(values, indices, k+1, divisor=divisor, max_size=max_size)


# Class for constructing a ball tree.
class BallTree:
    # Given points and a leaf size, construct a ball tree.
    def __init__(self, points=None, transpose=True, build=True,
                 leaf_size=1, num_threads=None, max_levels=None,
                 max_copy_bytes=None, **build_kwargs):
        # Set the internals.
        self.leaf_size = leaf_size
        self.built = 0
        self.size = 0
        self.tree = None
        self.usage = None
        self.sq_sums = None
        self.order = None
        self.radii = None
        self.medians = None
        # Assign the various pruning functions.
        self._balltree = ball_tree
        self._inner = prune.inner
        self._outer = prune.outer
        self._top = prune.top
        # Configure the OpenMP environment.
        self._balltree.configure(num_threads, max_levels)
        # Set the maximum number of bytes 
        if (max_copy_bytes is not None):
            self._balltree.max_copy_bytes = max_copy_bytes
        # Assign the points and build the tree, if provided.
        if (points is not None):
            self.add(points, transpose=transpose)
            if (build): self.build(**build_kwargs)


    # Add points to this ball tree, if the type is not yet defined,
    # then initialize the type of this tree to be same as added points.
    def add(self, points, transpose=True):
        # Transpose the points if the are not already column-vectors.
        if transpose: points = points.T
        # If there are existing points, make sure the new ones match.
        if (self.tree is not None):
            # If possible, pack the points into available space.
            if (points.shape[1] <= self.tree.shape[1] - self.size):
                self.tree[:,self.size:self.size+points.shape[1]] = points
                self.size += points.shape[1]
            else:
                # Store the old internal tree-defining attributes.
                old_tree = self.tree[:,:self.size]
                old_usage = self.usage[:self.size]
                old_sq_sums = self.sq_sums[:self.size]
                old_order = self.order[:self.size]
                old_radii = self.radii[:self.built]
                old_medians = self.medians[:self.built]
                # Compute the new size and initialize the memory for the new tree.
                to_allocate = max(2 * self.size, self.size + points.shape[1])
                self.size += points.shape[1]
                self.tree = np.zeros((self.tree.shape[0], to_allocate),
                                     dtype='float32', order='F')
                # Assign the relevant internals for this tree.
                self.usage = np.zeros(self.tree.shape[1], dtype='int64')
                self.sq_sums = np.zeros(self.tree.shape[1], dtype='float32')
                self.order = np.arange(self.tree.shape[1], dtype='int64') + 1
                self.radii = np.zeros(self.tree.shape[1], dtype='float32')
                self.medians = np.zeros(self.tree.shape[1], dtype='float32')
                # Pack the old points and the new points into a single tree
                #  while ensuring the built parts of the previous tree are still valid.
                self.tree[:,:old_tree.shape[1]] = old_tree
                self.tree[:,old_tree.shape[1]:self.size] = points
                self.usage[:old_usage.size] = old_usage
                self.sq_sums[:old_sq_sums.size] = old_sq_sums
                self.order[:old_order.size] = old_order
                self.radii[:old_radii.size] = old_radii
                self.medians[:old_medians.size] = old_medians
                # Delete the old tree components.
                del old_tree, old_usage, old_sq_sums, old_order, old_radii, old_medians
        else:
            # Save the points internally as the tree.
            self.tree = np.asarray(points, order='F', dtype='float32')
            # Assign the relevant internals for this tree.
            self.usage = np.zeros(self.tree.shape[1], dtype='int64')
            self.sq_sums = np.zeros(self.tree.shape[1], dtype='float32')
            self.order = np.arange(1, self.tree.shape[1]+1, dtype='int64')
            self.radii = np.zeros(self.tree.shape[1], dtype='float32')
            self.medians = np.zeros(self.tree.shape[1], dtype='float32')
            # Store BallTree internals for knowing how to evaluate.
            self.built = 0
            self.size = self.tree.shape[1]
        # Compute the square sums for the new points.
        self._balltree.compute_square_sums(
            self.tree[:,self.built:self.size],
            self.sq_sums[self.built:self.size]
        )

    # Restructure the ball tree so the points are in locally
    # contiguous blocks of memory (local by branch + leaf).
    def reorder(self):
        self._balltree.fix_order(self.tree, self.sq_sums, self.radii, self.medians, self.order[:self.built])

    # Build a tree out.
    def build(self, leaf_size=None, root=None, reorder=None):
        # Get the leaf size if it was not given.
        if (leaf_size is not None): self.leaf_size = leaf_size
        # Translate the root from python index to fortran index if provided.
        if (root is not None): root += 1
        # After this function completes, all points will be built.
        self.built = self.size
        # Only call the routines if the tree has positive size.
        if (self.size > 0):
            # Build tree (in-place operation).
            #    .tree     will not be modified
            #    .sq_sums  will contain the squared sums of each point
            #    .radii    will be modified to have the radius of specific
            #              node, or 0.0 if this point is a leaf.
            #    .medians  will be modified to have the distance to the
            #              median child node, or 0.0 if this point is a leaf.
            #    .order    will be the list of indices (1-indexed) that
            #              determine the structure of the ball tree.
            sq_dists = np.zeros(self.size, dtype='float32')
            self._balltree.build_tree(
                self.tree, self.sq_sums,self.radii, self.medians, sq_dists,
                self.order[:self.size], leaf_size=self.leaf_size, root=root,
            )
            # Restructure the ball tree so the points are in locally contiguous blocks of
            # memory (local by branch + leaf), as long as allowed (by user or memory).
            if (reorder or ((reorder is None) and (self.tree.nbytes < self._balltree.max_copy_bytes))):
                self._balltree.fix_order(self.tree, self.sq_sums, self.radii, self.medians, self.order[:self.built])

    # Find the "k" nearest neighbors to all points in z.
    def nearest(self, z, k=1, return_distance=True, transpose=True,
                budget=None, randomness=None, include_unbuilt=True):
        # If only a single point was given, convert it to a matrix.
        if (len(z.shape) == 1):
            singleton = True
            z = z.reshape((1,-1))
        else:
            singleton = False
        # Transpose the points if appropriate.
        if transpose: z = z.T
        # Make sure the 'k' value isn't bigger than the tree size.
        k = min(k, self.size)
        n = z.shape[1]
        # Initialize holders for output.
        points = np.asarray(z, order='F', dtype='float32')
        indices = np.ones((k, points.shape[1]), order='F', dtype='int64')
        dists = np.ones((k, points.shape[1]), order='F', dtype='float32')
        iwork = np.ones((k+self.leaf_size+2, min(n,self._balltree.number_of_threads)), order='F', dtype='int64')
        rwork = np.ones((k+self.leaf_size+2, min(n,self._balltree.number_of_threads)), order='F', dtype='float32')
        # Compute the nearest neighbors.
        self._balltree.nearest(
            points, k,
            self.tree, self.sq_sums, self.radii, self.medians, self.order[:self.built],
            self.leaf_size, indices, dists, iwork, rwork, to_search=budget, randomness=randomness
        )
        del iwork, rwork
        if (budget is not None):
            budget = max(0, budget - self.built)
        # Compute distance to points that have not been built into the tree.
        if (self.size > self.built):
            uindices = np.zeros((k, points.shape[1]), order='F', dtype='int64')
            udists = np.zeros((k, points.shape[1]), order='F', dtype='float32')
            uorder = np.arange(self.size-self.built, dtype='int64') + 1
            uleaf = self.size - self.built # The "leaf size" for the unbuilt points.
            iwork = np.ones((k+uleaf+2, min(n,self._balltree.number_of_threads)), order='F', dtype='int64')
            rwork = np.ones((k+uleaf+2, min(n,self._balltree.number_of_threads)), order='F', dtype='float32')
            self._balltree.nearest(
                points, k,
                self.tree[:,self.built:], self.sq_sums[self.built:], self.radii, self.medians,
                uorder, uleaf, uindices, udists, iwork, rwork, to_search=budget,
            )
            del iwork, rwork
            # Keep the nearest of the provided dists and indices.
            indices = np.concatenate((indices, uindices+self.built), axis=0)
            dists = np.concatenate((dists, udists), axis=0)
            to_keep = np.argsort(dists, axis=0)
            for i in range(points.shape[1]):
                indices[:k,i] = indices[to_keep[:k,i],i]
                dists[:k,i] = dists[to_keep[:k,i],i]
            # Only keep the top k.
            indices = indices[:k,:]
            dists = dists[:k,:]
        # Update the usage statistics for all points referenced in return values.
        self._balltree.bincount(indices.flatten(), self.usage)
        # Revert to singleton if that's what was provided.
        if singleton:
            dists = dists[:,0]
            indices = indices[:,0]
        # Return the results.
        if return_distance:
            if transpose: return dists.T, indices.T - 1
            else:         return dists,   indices - 1
        else:
            if transpose: return indices.T - 1
            else:         return indices - 1

    # Find the "k" nearest neighbors to all points in z. Uses the same
    # interface as the "BallTree.nearest" function, see help for more info.
    def query(self, *args, **kwargs): return self.nearest(*args, **kwargs)
    def __call__(self, *args, **kwargs): return self.nearest(*args, **kwargs)

    # Summarize this tree.
    def __str__(self):
        if not hasattr(self,"tree"): return "empty BallTree"
        return f"BallTree {self.tree.shape[::-1]}[:{self.size}] -- {self.built} built"

    # Return the usable length of this tree.
    def __len__(self):
        return self.size

    # Get an index point from the tree.
    def __getitem__(self, index):
        if (self.tree is not None):
            return self.tree[:,self.order[:self.built][index]-1]
        else:
            raise(IndexError("Attempting to __getitem__ with '[]' from an empty tree."))

    # Prune this tree and compact its points into the front of the
    # array, adjust '.size' and '.built' accordingly.
    def prune(self, levels=1, full=True, build=True, method="root"):
        assert(levels >= 1)
        assert(len(self) > 0)
        # Build the tree out if it is not fully built.
        if (full and (not (self.built == self.size))): self.build()
        # Get the size of the built portion of the tree.
        size = self.built
        # Compute the indices of the inner children (50th percentile) points.
        if (levels == 1):
            # Use the middle child for 1 level.
            indices = np.array([1], dtype=np.int64)
        else:
            # Handle the different methods of pruning the tree.
            if method in {"inner", "outer"}:
                to_keep = min(size, 1 + 2**(levels-1))
                # Otherwise, get the root, first outer, and all inners.
                indices = np.zeros(to_keep, dtype=np.int64)
                # Get the indices of the inner children of the built tree.
                if method == "inner":
                    indices[0] = 1
                    indices[1] = min((size + 2) // 2, size-1) + 1
                    self._inner(size, levels, indices[2:])
                elif method == "outer":
                    indices[0] = 1
                    indices[1] = 2
                    self._outer(size, levels, indices[2:])
            else:
                to_keep = min(size, 2**levels - 1)
                # Simply grab the root of the tree.
                indices = np.zeros(to_keep, dtype=np.int64)
                self._top(size, levels, indices)
            indices[:] -= 1
        # Stop this operation if the tree will remain unchanged.
        if (len(indices) == size): return
        # Adjust the recorded size and amount built of this tree.
        self.size = len(indices)
        self.built = 1
        # Keep those indices only.
        self.order[:self.size] = self.order[indices]
        # Rebuild the tree if desired.
        if build: self.build(root=0)


if __name__ == "__main__":
    print("_"*70)
    print(" TESTING BALL TREE MODULE")

    from tlux.plot import Plot
    from tlux.random import well_spaced_ball, well_spaced_box

    # A function for testing approximation algorithms.
    def f(x):
        x = x.reshape((-1,2))
        x, y = x[:,0], x[:,1]
        return (3*x + np.cos(8*x)/2 + np.sin(5*y))

    # Set the number of points, dimension, and random seed for repeatability.
    n = 200
    d = 2
    seed = 0
    np.random.seed(seed)

    # Create the test plot.
    x = np.asarray(well_spaced_box(n, d), order="C")
    y = f(x)

    # Create the tree.
    tree = BallTree(x, reorder=False)

    # Define a function that reproduces the function via application of nearest neighbor.
    fy = lambda x: y[tree(x, return_distance=False, k=1, budget=200)]
    x_min_max = np.asarray([np.min(x,axis=0), np.max(x,axis=0)]).T

    # Generate a plot to visualize the result.
    p = Plot()
    p.add("data", *(x.T), y)
    p.add_function("balltree", fy, *x_min_max, plot_points=5000, vectorized=True)
    p.show()
