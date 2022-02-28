import os

# Try using fmodpy to construct a Fortran wrapper over the `fmath.f90` library.
try:
    import fmodpy

    # Import the Fortran math library.
    _here = os.path.dirname(os.path.abspath(__file__))
    _fmath_path = os.path.join(_here, "fmath.f90")
    _fmath = fmodpy.fimport(_fmath_path, lapack=True, verbose=False, output_dir=_here)

    # Create local names for the internal functions.
    orthogonalize = _fmath.orthogonalize
    svd = _fmath.svd

# If anything above fails, fall back to Python implementations.
except ImportError:
    import numpy as np

    # Reorder by magnitude, orthogonalize, and normalize vectors in v.
    def orthogonalize(row_vecs, reorder=False, normalize=False):
        # Reorder so that the largest norm row vectors come first.
        if reorder:
            i = 0
            mags = np.linalg.norm(row_vecs[i:], axis=1)
            j = i + np.argmax(mags)
            row_vecs[[i,j],:] = row_vecs[[j,i],:]
        # Orthogonalize by iteratively removing current vector component from all others.
        vec_norm = np.linalg.norm(row_vecs[0])
        if (vec_norm > 0): row_vecs /= vec_norm
        for i in range(1, min(row_vecs.shape)):
            if reorder:
                mags = np.linalg.norm(row_vecs[i:], axis=1)
                j = i+np.argmax(mags)
                row_vecs[[j,i],:] = row_vecs[[i,j],:]
            row_vecs[i:] = (row_vecs[i:].T - np.dot(row_vecs[i:],row_vecs[i-1])
                            * row_vecs[i-1][:,None]).T
            if (normalize):
                vec_norm = np.linalg.norm(row_vecs[i])
                if (vec_norm > 0): row_vecs[i:] /= vec_norm
        # Return the now-orthogonal row vectors.
        return row_vecs

    # Compute the SVD by power iteration.
    def svd(X, steps=2):
        n, m = X.shape
        k = min(n, m)
        if (n >= m):
            A = np.matmul(X.T, X)
        else:
            A = np.matmul(X, X.T)
        # Get initial V.
        Vt = orthogonalize(A.copy(), reorder=True, normalize=True)
        # Do power iteration.a
        for i in range(steps):
            old_Vt = Vt
            Vt = orthogonalize(np.matmul(Vt, A), reorder=True, normalize=True)
            if (np.linalg.norm(Vt - old_Vt) < 2**(-50)): break
        # Compute singular values and their inverse.
        s = np.linalg.norm(np.matmul(A, Vt.T), axis=0)
        s[s!=0] = s[s!=0]**(1/2)
        # Compute the inverse of the nonzero singular values (for finishing SVD).
        si = np.where(s != 0, 1.0 / np.where(s != 0, s, 1.0), 0.0)
        # Compute left and right singular vectors.
        if (n >= m):
            U = np.matmul(np.matmul(X, Vt), np.diag(si))
            V = Vt.T
        else:
            V = np.matmul(np.matmul(np.diag(si), Vt), X)
            U = Vt.T
        return U, s, V
