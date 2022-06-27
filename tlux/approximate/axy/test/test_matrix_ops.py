import numpy as np
import fmodpy

# Matrix operations module.
mops = fmodpy.fimport("../matrix_operations.f90", blas=True, lapack=True).matrix_operations


# --------------------------------------------------------------------
#                        RANDOM_UNIT_VECTORS

# Generate test data.
n = 100 # number of points
d = 2   # dimenion of points

bd = 2 # full number of outputs
br = 1 # reduced rank approximation

# Random unit vectors.
a = np.zeros((n,d), dtype="float32")
mops.random_unit_vectors(a.T)

# Generate target outptus.
x = np.zeros((bd,d), dtype="float32").T
mops.random_unit_vectors(x)
b = np.array(np.matmul(a, x), order="F", dtype="float32")

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

# --------------------------------------------------------------------
#                         RADIALIZE

# TODO: Formalize this test.
x = np.asarray([
    [-0.3, 10.0],
    [-0.1, 10.0],
    [-0.2, 20.0],
    [-0.2, 00.0]
]).astype("float32").T

shift = np.zeros(x.shape[0], dtype="float32")
vecs = np.zeros((x.shape[0],)*2, dtype="float32", order="F")

a, b, c = mops.radialize(x, shift, vecs)

print("Test RADIALIZE (incomplete)")
print(a)
print()
print(b)
print()
print(c)
exit()


# --------------------------------------------------------------------
#                         ORTHOGONALIZE

# Test orthogonalization.
d = 4
m = np.random.random(size=(d,d)).astype("float32")
l = np.zeros(d, dtype="float32")

print("m: ")
print(m)
print(m.T @ m)
print(m @ m.T)
print()

mops.orthogonalize(m.T, l)

print("m: ")
print(m)
print(m.T @ m)
print(m @ m.T)
print()

print("l: ")
print(l)
print()

exit()
