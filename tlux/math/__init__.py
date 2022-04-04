from tlux.math.fraction import Fraction
from tlux.math.polynomial import Spline, Polynomial, NewtonPolynomial
from tlux.math.polynomial import fit as fit_spline
from tlux.math.polynomial import polynomial as fit_polynomial

# # Try using fmodpy to construct a Fortran wrapper over the `fmath.f90` library.
# try:
#     from .fmath import orthgonolize, svd
# except ImportError:
#     pass

# If anything above fails, fall back to Python implementations.
import numpy as np

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
def svd(row_vecs, steps=5, bias=1.0):
    dim = row_vecs.shape[1]
    # Initialize a holder for the singular valeus and the right
    #   singular vectors (the principal components).
    # Rescale the data for stability control, exit early when given only zeros.
    multiplier = abs(row_vecs).max()
    assert (multiplier > 0), "ERROR: Provided 'row_vecs' had no nonzero entries."
    multiplier = bias / multiplier
    # Compute the covariance matrix (usually the most expensive operation).
    covariance = np.matmul(multiplier*row_vecs.T, multiplier*row_vecs)
    # Compute the initial right singular vectors.
    right_col_vecs, lengths = orthogonalize(covariance.copy())
    # Do the power iteration.
    for i in range(steps):
        right_col_vecs, lengths = orthogonalize(
            np.matmul(covariance, right_col_vecs))
    # Compute the singular values from the lengths.
    singular_vals = lengths
    singular_vals[singular_vals > 0] = np.sqrt(
        singular_vals[singular_vals > 0]) / multiplier
    # Return the singular values and the right singular vectors.
    return singular_vals, right_col_vecs.T
