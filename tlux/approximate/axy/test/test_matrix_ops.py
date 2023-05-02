import numpy as np
import fmodpy

# Matrix operations module.
mops = fmodpy.fimport("../axy_matrix_operations.f90", blas=True, lapack=True).matrix_operations
print()


# --------------------------------------------------------------------
#                         ORTHOGONALIZE

# Tested up to ~400 dimension, where it fails witih precision=5.
def _test_orthogonalize(max_dimension=64, trials=100, precision=5):
    print("orthogonalize..", end=" ", flush=True)
    # Test orthogonalization.
    np.set_printoptions(linewidth=1000)
    for dimension in range(1, max_dimension+1):
        for trial in range(trials):
            np.random.seed(trial)
            matrix = np.random.random(size=(dimension,dimension)).astype("float32")
            lengths = np.zeros(dimension, dtype="float32")
            ortho = matrix.copy()
            mops.orthogonalize(ortho.T, lengths)
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
                f"mops.orthogonalize(ortho.T, lengths)\n" \
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
                f"mops.orthogonalize(ortho.T, lengths)\n" \
                f"bad_result = ortho @ ortho.T"
    print("passed.", flush=True)

_test_orthogonalize()


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
    return np.asarray(data @ rotation), center, variance, rotation

# Test function for visually checking the "radialize" function.
def _test_radialize(show=True):
    print("radialize..", end=" ", flush=True)
    # Generate data to test with.
    num_trials = 10
    np.random.seed(0)
    num_points = 1000
    dimension = 3
    from tlux.plot import Plot
    p = Plot()
    for i in range(num_trials):
        # Generate random data (that is skewed and placed off center).
        x, shift, scale, rotation = random_data(num_points, dimension)
        p.add(str(i+1), *x.T, marker_size=3)

        # Generate the radialized version.
        shift = np.zeros(dimension, dtype="float32")
        transform = np.zeros((dimension, dimension), dtype="float32", order="F")
        inverse = np.zeros((dimension, dimension), dtype="float32", order="F")
        to_flatten = (i % 2) == 0
        descriptor = 'radialized' if to_flatten else 'normalized'
        xr, shift, transform, inverse = mops.radialize(
            x=(x.copy()).T, shift=shift, vecs=transform, inverse=inverse,
            max_to_flatten=(0 if to_flatten else None),
        )
        p.add(f"{i+1} {descriptor}", *xr, marker_size=3)

        # Use the inverse to "fix" the data back to its original form.
        xf = (inverse.T @ xr).T
        xf -= shift
        p.add(f"{i+1} fixed", *xf.T, marker_size=3)

        # Use the provided shift and transform to repeate the radialization process.
        xrr = (x + shift) @ transform
        p.add(f"{i+1} re {descriptor}", *xrr.T, marker_size=3)
    if (show): p.show()
    print("passed.", flush=True)

_test_radialize(show=False)


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


# _test_least_squares()
