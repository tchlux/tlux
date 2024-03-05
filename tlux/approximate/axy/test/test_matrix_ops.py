import numpy as np
import fmodpy

# Matrix operations module.
try:
    from axy_matrix_operations import matrix_operations as mops
except:
    mops = fmodpy.fimport("../axy_matrix_operations.f90", blas=True, lapack=True, omp=True).matrix_operations


# --------------------------------------------------------------------
#                         ORTHONORMALIZE

# Tested up to ~400 dimension, where it fails witih precision=5.
def _test_orthonormalize(max_dimension=256, trials=100, precision=5):
    print("orthonormalize..", flush=True)
    # Test orthogonalization.
    np.set_printoptions(linewidth=1000)
    import tqdm
    for dimension in tqdm.tqdm(range(1, max_dimension+1)):
        for trial in range(trials):
            np.random.seed(trial)
            matrix = np.random.random(size=(dimension,dimension)).astype("float32")
            lengths = np.zeros(dimension, dtype="float32")
            ortho = matrix.copy()
            mops.orthonormalize(ortho.T, lengths)
            if (min(lengths) == 0.0): continue
            identity = np.identity(dimension).astype("float32")
            AtA = ortho.T @ ortho
            assert (abs(AtA-identity).max()) < 10**(-precision), \
                "Results of (At A) are not orthogonal enough.\n" \
                f"  Dimension = {dimension}\n" \
                f"  Trial = {trial}\n\n" \
                f"{(AtA).round(precision)}\n\n" \
                f"matrix = np.asarray({matrix.tolist()}, dtype='float32')\n" \
                f"lengths = np.zeros({dimension}, dtype='float32')\n" \
                "ortho = matrix.copy()\n" \
                f"mops.orthonormalize(ortho.T, lengths)\n" \
                f"bad_result = ortho.T @ ortho"
            AAt = ortho @ ortho.T
            assert (abs(AAt-identity).max()) < 10**(-precision), \
                "Results of (A At) are not orthogonal enough.\n" \
                f"  Dimension = {dimension}\n" \
                f"  Trial = {trial}\n\n" \
                f"{(AAt).round(precision)}\n\n" \
                f"matrix = np.asarray({matrix.tolist()}, dtype='float32')\n" \
                f"lengths = np.zeros({dimension}, dtype='float32')\n" \
                "ortho = matrix.copy()\n" \
                f"mops.orthonormalize(ortho.T, lengths)\n" \
                f"bad_result = ortho @ ortho.T"
    print(" passed.", flush=True)


# --------------------------------------------------------------------
#                                  SVD

# Test the SVD function with matrices having more columns than rows.
def _test_svd(max_dimension=16, n=100, trials=100, precision=2):
    print("svd..", flush=True)
    np.set_printoptions(linewidth=1000)
    import tqdm

    allowed_errors = 1

    for dimension in tqdm.tqdm(range(1, max_dimension + 1)):
        for trial in range(trials):
            np.random.seed(trial)
            # Create a matrix with more columns than rows
            variances = np.random.random(size=(dimension if (trial % 2 == 0) else n,))
            matrix = np.random.normal(
                loc=0.0,
                scale=variances,
                size=((n, dimension) if (trial % 2 == 0) else (dimension, n)),
            ).astype("float32")
            if (dimension > 1):
                matrix -= matrix.mean(axis=0)
            matrix = matrix.T  # Make Fortran contiguous.
            # Prepare output containers for singular values and vectors
            singular_values = np.zeros(min(matrix.shape), dtype="float32", order="F")
            VT = np.zeros((dimension if (trial % 2 == 0) else n, min(matrix.shape)), dtype="float32", order="F")
            # Call the svd function
            s, vt, rank, _ = mops.svd(a=matrix, s=singular_values, vt=VT, rank=True, steps=1000)
            # Check basic truth.
            assert (rank > 0), f"Rank is 0. Should certainly be positive for matrix with any nonzero entries.\n" \
                f"matrix = np.asarray({matrix.tolist()}, dtype='float32')\n" \
                "singular_values = np.zeros(min(matrix.shape), dtype='float32')\n" \
                f"VT = np.zeros(({dimension}, min(matrix.shape)), dtype='float32')\n" \
                "mops.svd(matrix, singular_values, VT, rank=True)\n" \
            # Check that the singular values are nonincreasing.
            assert (np.all(s[:-1] >= s[1:])), f"Singular values are not in nonincreasing order (they are *increasing* and shouldn't be).\n  s = {s}"
            # Orthonormality may be degraded for higher dimension (that's okay), so only bother checking strictly for lower values.
            if (dimension <= 16):
                # Check if the singular vectors (columns of VT) are orthogonal
                if (VT.shape[0] <= VT.shape[1]):
                    VTV = VT @ VT.T
                else:
                    VTV = VT.T @ VT
                # 0-out the rows and columns associated with null vectors (rank deficient cases).
                for i in np.argsort(np.abs(np.diag(VTV) - 1))[rank:]:
                    VTV[i,:] = 0.0
                    VTV[:,i] = 0.0
                # Get the "kept" indices.
                diag_diff = np.abs(np.diag(VTV) - 1)
                diag_diff.sort()
                off_diag_diff = VTV - np.diag(np.diag(VTV))
                assert ((abs(off_diag_diff).max() < 10**(-precision))
                        and ((diag_diff[:rank]).max() < 10**(-precision))), \
                    "Singular vectors are not orthogonal enough.\n" \
                    f"  Dimension = {dimension}\n" \
                    f"  Trial = {trial}\n" \
                    f"  Rank = {rank}\n" \
                    f"  abs(off_diag_diff).max() = {abs(off_diag_diff).max()}\n" \
                    f"  (diag_diff[:rank]).max() = {(diag_diff[:rank]).max()}\n\n" \
                    f"{(VTV).round(precision)}\n\n" \
                    f"matrix = np.asarray({matrix.tolist()}, dtype='float32')\n" \
                    "singular_values = np.zeros(min(matrix.shape), dtype='float32')\n" \
                    f"VT = np.zeros(({dimension}, min(matrix.shape)), dtype='float32')\n" \
                    "mops.svd(matrix, singular_values, VT, rank=True)\n" \
                    "bad_result = VT @ VT.T"
            # Check that numpy agrees with the computed value of VT.
            U, S, Vh = np.linalg.svd(matrix.T, full_matrices=False)
            # Make sure the vectors are oriented the same directions.
            #  (this is not necessary for absolute cos similarity, but for debugging)
            for i in range(dimension):
                if (Vh[i,0] * vt[0,i] < 0):
                    Vh[i,:] *= -1
            # Compute the pairwise cosine similarities.
            cos_similarities = np.abs(np.sum(Vh.T * vt, axis=0))
            sim_inds = cos_similarities.argsort()
            assert (cos_similarities[sim_inds[-rank]] > 0.95), \
                "Computed Vt was incorrect.\n" \
                f"  Dimension = {dimension}\n" \
                f"  Trial = {trial}\n" \
                f"  Rank = {rank}\n\n" \
                f"{(vt).round(precision)}\n\n" \
                f"{(Vh.T).round(precision)}\n\n" \
                f"{cos_similarities}\n\n" \
                f"matrix = np.asarray({matrix.tolist()}, dtype='float32')\n" \
                "singular_values = np.zeros(min(matrix.shape), dtype='float32')\n" \
                f"VT = np.zeros(({dimension}, min(matrix.shape)), dtype='float32')\n" \
                "mops.svd(matrix, singular_values, VT, rank=True)\n" \
                "bad_result = VT @ VT.T"
            # Optionally, check the accuracy of singular values by reconstructing the matrix
            # This part is left as an exercise; it involves using singular values and vectors
            # to reconstruct the original matrix and comparing it with the input matrix.
    print(" passed.", flush=True)


# --------------------------------------------------------------------
#                         RADIALIZE

# Generate a random rotation matrix (that rotates a random amount along each axis).
def random_rotation(dimension):
    # Determine the order of the dimension rotations (done in pairs)
    rotation_order = np.arange(dimension)
    # Generate the rotation matrix by rotating two dimensions at a time.
    rotation_matrix = np.identity(dimension)
    for (d1, d2) in zip(rotation_order, np.roll(rotation_order,1)):
        next_rotation = np.identity(dimension)
        # Establish the rotation
        rotation = np.random.random() * 2 * np.pi
        next_rotation[d1,d1] =  np.cos(rotation)
        next_rotation[d2,d2] =  np.cos(rotation)
        next_rotation[d1,d2] =  np.sin(rotation)
        next_rotation[d2,d1] = -np.sin(rotation)
        # Compound the paired rotations
        rotation_matrix = np.matmul(next_rotation, rotation_matrix)
        # When there are two dimenions or fewer, do not keep iterating.
        if (dimension <= 2): break
    return rotation_matrix

# Generate random data with skewed distribution along the principle axes.
def random_data(num_points, dimension, box=10, skew=lambda x: 1 * x**2 / sum(x**2)):
    center = (np.random.random(size=(dimension,)) * box - box/2).astype("float32")
    variance = skew(np.random.random(size=(dimension,))).astype("float32")
    data = np.random.normal(center, variance, size=(num_points,dimension)).astype("float32")
    rotation = random_rotation(dimension).astype("float32")
    return data, center, variance, rotation

# Test function for visually checking the "radialize" function.
def _test_radialize(show=True, num_points=1000, dimension=5, num_trials=1000, seed=0, precision=4):
    print("radialize..", flush=True)
    # Generate data to test with.
    np.random.seed(seed)
    # Plotting.
    show = show and (dimension in {2,3}) and (num_points <= 5000)
    if show:
        from tlux.plot import Plot
        p = Plot()
    # Trials.
    import tqdm
    for i in tqdm.tqdm(range(num_trials)):
        # Generate random data (that is skewed and placed off center).
        x, shifted_by, scaled_by, rotate_by = random_data(num_points, dimension)
        to_flatten = (i % 2) == 1
        iterative_computation = (i % 3) == 1
        iterative_computation = True
        # ************************************************************************************
        # Generate the radialized version.
        if (not iterative_computation):
            # Call the RADIALIZE routine once.
            shift = np.zeros(dimension, dtype="float32")
            transform = np.zeros((dimension, dimension), dtype="float32", order="F")
            inverse = np.zeros((dimension, dimension), dtype="float32", order="F")
            order = np.zeros(dimension, dtype="int32")
            xr, shift, transform, inverse, order = mops.radialize(
                x=(x @ rotate_by).T, shift=shift, vecs=transform, inverse=inverse, order=order,
                max_to_flatten=(None if to_flatten else 0),
                maxbound=(None if to_flatten else False), # Use an average bound when not flattening.
                svd_steps=10,
                update=False,
            )
        else:
            # Call the RADIALIZE routine iteratively, doing 1 SVD step each time.
            shift = np.zeros(dimension, dtype="float32")
            transform = np.zeros((dimension, dimension), dtype="float32", order="F")
            inverse = np.zeros((dimension, dimension), dtype="float32", order="F")
            order = np.zeros(dimension, dtype="int32")
            xr, shift, transform, inverse, order = mops.radialize(
                x=(x @ rotate_by).T, shift=shift, vecs=transform, inverse=inverse, order=order,
                max_to_flatten=(None if to_flatten else 0),
                maxbound=(None if to_flatten else False), # Use an average bound when not flattening.
                svd_steps=1,
                update=False,
            )
            # Single steps of SVD.
            steps = 40
            for s in range(steps):
                xr, shift, transform, inverse, order = mops.radialize(
                    x=(x @ rotate_by).T, shift=shift, vecs=transform, inverse=inverse, order=order,
                    max_to_flatten=(None if to_flatten else 0),
                    maxbound=(None if to_flatten else False), # Use an average bound when not flattening.
                    svd_steps=1,
                    update=True,
                )
        # ************************************************************************************
        # 
        # Use the inverse to "fix" the data back to its original form.
        xf = (inverse.T @ xr).T
        xf -= shift
        # Use the provided shift and transform to repeate the radialization process.
        xrr = (((x@rotate_by) + shift) @ transform).T
        # 
        # Plotting.
        if show:
            descriptor = 'radialized' if to_flatten else 'normalized'
            p.add(f"{i+1}", *(x@rotate_by).T, marker_size=3)
            p.add(f"{i+1} {descriptor}", *xr, marker_size=3)
            p.add(f"{i+1} fixed", *xf.T, marker_size=3)
            p.add(f"{i+1} re {descriptor}", *xrr, marker_size=3)
        # If the minimum norm of a vector is zero, then a component was lost due to
        #  it being too small / close to degenerate. Those reconstructions will have
        #  higher error standards, but are just skipped here (TODO: compute updated bound).
        min_norm = np.linalg.norm(inverse,axis=1).min()
        if (min_norm > 0):
            # Validate the accuracy of the transformations.
            assert (abs((x@rotate_by) - xf).mean() < 10**(-precision)), f"Error in recreated data (by applying the returned 'inverse' to normalized data) is too high."
            assert (abs(xr - xrr).mean() < 10**(-precision)), f"Error in renormalized data (by applying 'shift' and 'vecs' to the input data) is too high."
        else:
            # Validate the accuracy of the transformations.
            assert (abs((x@rotate_by) - xf).mean() < 0.0002), f"Error in recreated data (by applying the returned 'inverse' to normalized data) is too high."
            assert (abs(xr - xrr).mean() < 0.0002), f"Error in renormalized data (by applying 'shift' and 'vecs' to the input data) is too high."

        # Check that the mean-zero and componentwise scaling properties are maintained.
        shift_max_error = np.percentile(np.abs(xrr.mean(axis=1)), 0.95)
        assert (shift_max_error < 10**(-precision)), f"Shift maximum error is too high, {shift_max_error} > 10^({-precision})."
        to_keep = (np.linalg.norm(transform, axis=0) > 0)
        # Depending on whether data was flattened, either look at the average deviation of points.
        if (to_flatten):
            scale_max_error = 1-xrr[to_keep].std(axis=1)
            scale_max_error = np.max(np.abs(scale_max_error))
        else:
            scale_max_error = np.std(xrr[to_keep], axis=1)
            scale_max_error = abs(1.0 - scale_max_error.mean())
        assert (scale_max_error < 10**(-precision)), f"Scale maximum error is too high, {scale_max_error} > 10^({-precision})."        
        # 
        # INFO: Should we verify that the "correct" shift and rotation were discovered?
        #       Or is it sufficient to check that the data meets our desired properties?
        # 
    # Plotting.
    if (show): p.show(file_name="/tmp/test_matrix_ops_radialize.html", aspect_mode="data")
    # Finished.
    print(" passed.", flush=True)


# --------------------------------------------------------------------
#                            LEAST_SQUARES

def _test_least_squares():
    # Use the matrix operations subroutine to solve the least squares problem.
    br = 2
    d = 3
    mo_x = np.zeros((d,br), order="F", dtype="float32")
    mops.least_squares(
        bytes("T","ascii"),
        np.array(a.T, order="F", dtype="float32"),
        np.array(b, order="F", dtype="float32"),
        mo_x)

    np_x, residual = np.linalg.lstsq(a, b, rcond=None)[:2]

    print()
    print("x: ")
    print(x)

    print()
    print("np_x: ")
    print(np_x)

    print()
    print("mo_x: ")
    print(mo_x)

    print()
    print("mo_x (scaled to full output):")
    upscale = np.linalg.lstsq(np.matmul(a, mo_x), b, rcond=None)[0]
    print(np.matmul(mo_x, upscale))
    print()

    # Generate a visiaul.
    from tlux.plot import Plot
    p = Plot()
    p.add("a", *a.T, color=1)
    # Show the true values.
    for i in range(bd):
        p.add(f"b{i}", *a.T, b[:,i], color=2+i, shade=True)
    # Show the literal values produced by the mo regression.
    for i in range(br):
        p.add(f"app b{i}", *a.T, np.matmul(a, mo_x)[:,i],
              color=(0,0,0,0.8), marker_size=2)
    # Show the original approximations recovered from the mo regression.
    for i in range(bd):
        p.add(f"recon b{i}", *a.T, np.matmul(a, np.matmul(mo_x, upscale))[:,i],
              color=2+i, shade=True, marker_size=4, marker_line_width=1)

    # Show the plot, different depending on if it is 2D (points only) or 3D (with regression).
    if (bd > 0):
        p.show(x_range=[-1.1,1.1], y_range=[-1.1,1.1], z_range=[-1.1,1.1])
    else:
        p.show(x_range=[-2.3,2.3], y_range=[-1.1,1.1])
    print()


# --------------------------------------------------------------------

if __name__ == "__main__":
    # Passes up to 256 in ~1.3 minutes.
    _test_orthonormalize(
        max_dimension=128,
        trials=100
    ) 
    # Passes up to 16 in ~13 seconds.
    _test_svd()
    # Passes up to 32 in ~5 seconds.
    _test_radialize(
        show=True, # "show" only goes into effect when dimension <= 3
        dimension=32,
        num_points=1000,
        num_trials=100
    )
    # TODO: Need to rework the least squares function.
    # _test_least_squares()
