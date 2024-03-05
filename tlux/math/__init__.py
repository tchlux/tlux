# If anything above fails, fall back to Python implementations.
import numpy as np

from tlux.math.fraction import Fraction
from tlux.math.polynomial import Spline, Polynomial, NewtonPolynomial
from tlux.math.polynomial import fit as fit_spline
from tlux.math.polynomial import polynomial as fit_polynomial


# Given column vectors (in a 2D numpy array), orthogonalize and return
#  the orthonormal vectors and the lengths of the orthogonal components.
def orthogonalize(col_vecs, essentially_zero=2**(-26)):
    rank = 0
    lengths = np.zeros(col_vecs.shape[1])
    for i in range(col_vecs.shape[1]):
        lengths[i:] = np.linalg.norm(col_vecs[:,i:], axis=0)
        # Move the longest vectors to the front / leftmost position (pivot).
        descending_order = np.argsort(-lengths[i:])
        lengths[i:] = lengths[i:][descending_order]
        col_vecs[:,i:] = col_vecs[:,i+descending_order]
        # Break if none of the remaining vectors have positive length.
        if (lengths[i] < essentially_zero): break
        col_vecs[:,i] /= lengths[i]
        # Orthgonolize, remove vector i from all remaining vectors.
        if (i+1 < col_vecs.shape[1]):
            v = np.dot(col_vecs[:,i],col_vecs[:,i+1:]) * col_vecs[:,i:i+1]
            col_vecs[:,i+1:] -= v
    # Return the orthonormalized vectors and their lengths (before normalization).
    return col_vecs, lengths


# Compute the singular values and the right singular vectors for a matrix of row vectors.
def pca(row_vecs, steps=5, bias=1.0, nvecs=None):
    # Make sure the row vectors are a numpy array.
    row_vecs = np.asarray(row_vecs)
    if (len(row_vecs.shape) == 1):
        row_vecs = row_vecs[None,:]
    # Get the dimension of the vectors.
    dim = row_vecs.shape[1]
    # Handle zero-sized inputs by returning zero-sized outputs.
    if (row_vecs.size == 0):
        return np.zeros((0,), dtype=float), np.zeros((0,dim), dtype=float)
    # Set the number of vectors to compute (if that wasn't provided).
    if (nvecs is None):
        nvecs = row_vecs.shape[1]
    nvecs = min(row_vecs.shape[1], max(1, nvecs))
    # Initialize a holder for the singular values and the right
    #   singular vectors (the principal components).
    # Rescale the data for stability control, exit early when given only zeros.
    multiplier = max(abs(np.nanmax(row_vecs)), abs(np.nanmin(row_vecs)))
    # Handle the case of 0-variance data.
    if (multiplier <= 0):
        singular_vals = np.zeros(row_vecs.shape[1])
        right_col_vecs = np.zeros((row_vecs.shape[1], nvecs))
        return singular_vals, right_col_vecs
    # If there are any nan values, replace them with the mean.
    elif (np.any(np.isnan(row_vecs))):
        mean = np.nanmean(row_vecs, axis=0)
        row_vecs = np.where(
            np.isnan(row_vecs),
            mean,
            row_vecs
        )
    # Create the multiplier for rescaling (for stability).
    multiplier = bias / multiplier
    # Compute the covariance matrix (usually the most expensive operation).
    row_vecs = multiplier * row_vecs
    center = row_vecs.mean(axis=0)
    row_vecs -= center
    # Method (1) precomputes the covariance matrix and then does power iterations.
    if (row_vecs.shape[0] >= row_vecs.shape[1]):
        # Compute the initial right singular vectors.
        covariance = np.matmul(row_vecs.T, row_vecs)
        right_col_vecs, lengths = orthogonalize(covariance.copy())
        for i in range(steps):
            right_col_vecs, lengths = orthogonalize(np.matmul(covariance, right_col_vecs))
    # 
    # Method (2) starts with random vectors does power iterations with the implicit covariance matrix.
    else:
        right_col_vecs, lengths = orthogonalize( # Add 2 for error buffer & increased convergence.
            np.random.normal(size=(row_vecs.shape[1], min(nvecs+2, row_vecs.shape[1])))
        )
        for i in range(steps):
            right_col_vecs, lengths = orthogonalize(
                np.matmul(row_vecs.T, np.matmul(row_vecs, right_col_vecs))
            )
    # Compute the singular values from the lengths.
    singular_vals = lengths
    singular_vals[singular_vals > 0] = np.sqrt(
        singular_vals[singular_vals > 0]) / multiplier
    # Return the singular values and the right singular vectors.
    return singular_vals[:nvecs], right_col_vecs.T[:nvecs]


# Use the SVD routine to project data onto its first N principal components.
def project(row_vecs, dim, **pca_kwargs):
    pca_kwargs["nvecs"] = dim
    # Compute the principal components via a singular value decomposition.
    singular_values, projection_vecs = pca(row_vecs, **pca_kwargs)
    row_vecs = row_vecs @ projection_vecs[:dim,:].T
    # Add 0's to the end if the projection was not enough.
    if (dim > row_vecs.shape[1]):
        row_vecs = np.concatenate((row_vecs, np.zeros((row_vecs.shape[0], dim - row_vecs.shape[1]))), axis=1)
    return row_vecs


# Given "d" categories that need to be converted into real space,
# generate a regular simplex in (d-1)-dimensional space. (all points
# are equally spaced from each other and the origin) This process
# guarantees that all points are not placed on a sub-dimensional
# manifold (as opposed to a ones encoding).
def regular_simplex(d, volume=None):
    import numpy as np
    # Special cases for one and two categories
    if ((not isinstance(d, int)) or (d <= 1)):
        class InvalidDimension(Exception): pass
        raise(InvalidDimension(f"Number of points 'd' must be an integer greater than 1. Received '{type(d).__name__} {d}'"))
    elif d == 2: return np.asarray([[0.],[1.]], dtype="float32")
    # Initialize all points to be zeros.
    points = np.zeros((d,d-1), dtype="float32")
    # Set the initial first point as 1
    points[0,0] = 1
    # Calculate all the intermediate points
    for i in range(1,d-1):
        # Set all points to be flipped from previous calculation while
        # maintaining the angle "arcos(-1/d)" between vectors.
        points[i:,i-1] = -1/(d-i) * points[i-1,i-1]
        # Compute the new coordinate using pythagorean theorem
        points[i,i] = (1 - sum(points[i,:i]**2))**(1/2)
    # Set the last coordinate of the last point as the negation of the previous
    points[i+1,i] = -points[i,i]
    # Normalize the edge lengths to all be 1.
    points *= (2 + 2/(d-1))**(-1/2)
    # If no volume was specified, return the vertices of the regular simplex.
    if volume is None: return points
    elif (volume > 0):
        # Return a regular simplex with volume specified by first converting
        # to unit edge length, then using the formula from [1]:
        # 
        #   volume  =   (length^d / d!) ([d+1] / 2^d)^(1/2)
        # 
        # where "volume" is the volume of the simplex defined by d+1
        # vertices, "d" is the dimension the simplex lives in (number
        # of vertices minus one), and "length" is the length of all
        # edges between vertices of the simplex. This gives:
        # 
        #  length  =  [ volume d! ([d+1] / 2^d)^(-1/2) ]^(1/d)
        #          =  (volume d!)^(1/d) ([d+1] / 2^d)^(-1/(2d))
        # 
        # Reference:
        #  [1] https://www.researchgate.net/publication/267127613_The_Volume_of_an_n-Simplex_with_Many_Equal_Edges
        # 
        from math import factorial
        d -= 1
        edge_length = (volume*factorial(d))**(1/d) * ((d+1) / 2**d)**(-1/(2*d))
        return points * edge_length


# Define a custom error for passing arrays of the wrong shape to pairwise.
class ShapeError(Exception): pass

# Compute the distance between all pairs of elements in a single list,
# or the distance of all combined pairs between two lists.
def pairwise_distance(x1, x2=None, diag=float('inf')):
    # Define an error for having the wrong array shape.
    if (len(x1.shape) != 2): raise(ShapeError("Only 2D NumPy arrays are allowed."))
    # Determine use case.
    if (x2 is None):
        # Compute the pairwise distances.
        x1_sq = np.sum(x1**2, axis=1, keepdims=True)
        d = (x1_sq + x1_sq[:,0]) - 2 * np.matmul(x1, x1.T)
        # Protect against the errors thta will occur along the diagonal.
        d[np.diag_indices(len(d))] = 1.0
        d[:,:] = np.sqrt(d[:,:])
        d[np.diag_indices(len(d))] = diag
        return d
    else:
        if (len(x2.shape) != 2): raise(ShapeError("Only 2D NumPy arrays are allowed."))
        # Compute the pairwise distance between memebers of each set.
        x1_sq = np.sum(x1**2, axis=1, keepdims=True)
        x2_sq = np.sum(x2**2, axis=1)
        return np.sqrt(x1_sq + x2_sq - 2 * np.matmul(x1, x2.T))

