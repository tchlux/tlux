import numpy as np
import fmodpy

# Matrix operations module.
mops = fmodpy.fimport("../axy_matrix_operations.f90", blas=True, lapack=True).matrix_operations

# --------------------------------------------------------------------
#                         ORTHONORMALIZE

# Tested up to ~400 dimension, where it fails witih precision=5.
def _test_orthonormalize(max_dimension=256, trials=100, precision=5):
    print("orthonormalize..", end=" ", flush=True)
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
    print("passed.", flush=True)


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
    print("radialize..", end=" ", flush=True)
    print()
    # Generate data to test with.
    np.random.seed(seed)
    # Plotting.
    show = show and (dimension in {2,3}) and (num_points <= 5000)
    if show:
        from tlux.plot import Plot
        p = Plot()
    # Trials.
    for i in range(num_trials):
        # Generate random data (that is skewed and placed off center).
        x, shifted_by, scaled_by, rotate_by = random_data(num_points, dimension)
        # Generate the radialized version.
        shift = np.zeros(dimension, dtype="float32")
        transform = np.zeros((dimension, dimension), dtype="float32", order="F")
        inverse = np.zeros((dimension, dimension), dtype="float32", order="F")
        order = np.zeros(dimension, dtype="int32")
        to_flatten = (i % 2) == 1
        xr, shift, transform, inverse, order = mops.radialize(
            x=(x @ rotate_by).T, shift=shift, vecs=transform, inverse=inverse, order=order,
            max_to_flatten=(None if to_flatten else 0),
            maxbound=(None if to_flatten else False), # Use an average bound when not flattening.
        # 
            svd_steps=1,
            svd_update=False,
        )
        # Debugging values.
        scale_after_rotation = np.linalg.norm((np.identity(dimension) * scaled_by) @ rotate_by, axis=0)
        value_norm = (xr**2).mean()
        # Single steps of SVD.
        order.sort()
        steps = 10
        for s in range(steps):
            xr, shift, transform, inverse, order = mops.radialize(
                x=(x @ rotate_by).T, shift=shift, vecs=transform, inverse=inverse, order=order,
                max_to_flatten=(None if to_flatten else 0),
                maxbound=(None if to_flatten else False), # Use an average bound when not flattening.
                svd_steps=1, # TODO: Doing more outer steps should be equivalent, but this is dominating.
                svd_update=True,
            )
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
            print("test_matrix_ops.py Line 148: ", flush=True)
            scale_max_error = 1-xrr[to_keep].std(axis=1)
            scale_max_error = np.max(np.abs(scale_max_error))
        else:
            print("test_matrix_ops.py Line 152: ", flush=True)
            scale_max_error = np.std(xrr[to_keep], axis=1)
            scale_max_error = abs(1.0 - scale_max_error.mean())
        print("scale_max_error: ", scale_max_error, flush=True)
        assert (scale_max_error < 10**(-precision)), f"Scale maximum error is too high, {scale_max_error} > 10^({-precision})."        
        # 
        # INFO: Should we verify that the "correct" shift and rotation were discovered?
        #       Or is it sufficient to check that the data meets our desired properties?
        # 
    # Plotting.
    if (show): p.show(file_name="/tmp/test_matrix_ops_radialize.html", aspect_mode="data")
    # Finished.
    print("passed.", flush=True)


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


if __name__ == "__main__":
    # _test_orthonormalize(max_dimension=64, trials=100)  # Passes up to 256 in ~2 minutes.
    _test_radialize(
        show=True, # "show" only goes into effect when dimension <= 3
        dimension=6,
        num_trials=2
    )
    # _test_least_squares()



# TODO:
# 
# The magnitude appears to be incorrect when radializing with one SVD step.
# 
