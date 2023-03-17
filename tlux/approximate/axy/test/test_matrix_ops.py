import numpy as np
import fmodpy

# Matrix operations module.
rand = fmodpy.fimport("../random.f90", name="fortran_random",
                      blas=True, lapack=True).random
mops = fmodpy.fimport("../matrix_operations.f90", name="fortran_matrix_operations",
                      blas=True, lapack=True).matrix_operations


# Make sure that the random integer generation has desired behavior.
def _test_random_integer():
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

_test_random_integer()


# --------------------------------------------------------------------
#                            STABLE_MEAN

def _test_stable_mean():
    from tlux.math import Fraction
    shift = 1e11
    scale = 1e3
    n = 100000
    d = 4
    # Generate data that has a mean and is offset from the origin.
    target_mean = ((np.random.random(size=(d,)) - 0.5)*2).astype("float32")
    matrix = np.random.normal(target_mean, size=(n,d)).astype("float32")
    print("matrix: ", matrix)
    matrix = matrix * scale + shift
    target_mean = target_mean * scale + shift
    print(matrix)
    # Compute the "true mean" with exact arithmetic.
    exact_transpose = [[Fraction(v) for v in col] for col in matrix.T]
    true_mean = np.asarray([sum(col) / len(col) for col in exact_transpose], dtype=object)
    # Compute the stable mean.
    mean = np.zeros((d,), dtype="float32")
    mops.stable_mean(matrix.T, mean, dim=2)
    print("true mean:     ", (true_mean.astype("float32")).tolist())
    print("computed mean: ", mean.tolist())
    print("difference:    ", (true_mean.astype("float32") - mean).tolist())
    print()

# _test_stable_mean()
# exit()


# --------------------------------------------------------------------
#                        RANDOM_UNIT_VECTORS

# TODO: Add test with large number of vectors, ensure no Inf or Nan generated.
def _test_random_unit_vectors():
    # Generate test data.
    n = 100 # number of points
    d = 3   # dimenion of points

    bd = 2 # full number of outputs
    br = 1 # reduced rank approximation

    # Random unit vectors.
    a = np.zeros((n,d), dtype="float32")
    rand.random_unit_vectors(a.T)

    from tlux.plot import Plot
    p = Plot()
    for i in range(n):
        l = np.vstack([[0]*d, a[i]])
        p.add(str(i), *l.T, mode="lines")
    p.show()

    # Generate target outptus.
    x = np.zeros((bd,d), dtype="float32").T
    rand.random_unit_vectors(x)
    b = np.array(np.matmul(a, x), order="F", dtype="float32")

_test_random_unit_vectors()
exit()


# --------------------------------------------------------------------
#                         ORTHOGONALIZE

def _test_orthogonalize(max_dimension=32, trials=1000, precision=4):
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


# matrix = np.asarray([[0.8701241612434387, 0.5822769403457642], [0.2788389325141907, 0.18591123819351196]], dtype='float32')
# lengths = np.zeros(2, dtype='float32')
# ortho = matrix.copy()
# mops.orthogonalize(ortho.T, lengths)
# bad_result = ortho.T @ ortho
# print()
# print(matrix)
# print()
# print(lengths)
# print()
# print(ortho)
# print()
# print(bad_result)
# print()
# exit()

# _test_orthogonalize()
# exit()

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
def _test_radialize():
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
            flatten=to_flatten
        )
        p.add(f"{i+1} {descriptor}", *xr, marker_size=3)

        # Use the inverse to "fix" the data back to its original form.
        xf = (inverse.T @ xr).T
        xf -= shift
        p.add(f"{i+1} fixed", *xf.T, marker_size=3)

        # Use the provided shift and transform to repeate the radialization process.
        xrr = (x + shift) @ transform
        p.add(f"{i+1} re {descriptor}", *xrr.T, marker_size=3)
    p.show()

_test_radialize()
exit()


# --------------------------------------------------------------------
#                            LEAST_SQUARES

# Use the matrix operations subroutine to solve the least squares problem.
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
