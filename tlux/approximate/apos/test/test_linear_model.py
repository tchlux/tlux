import time
import numpy as np
import fmodpy

# Import two different linear models.
from tlux.approximate.apos import APOS

# Create data.
np.random.seed(0)
n = 100
d = 6

# Create test data.
x = np.random.random(size=(n,d)).astype("float32")
y = np.cos(x * np.linspace(1,d,d))[:,:(d+1)//2].astype("float32")

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)

# Fit the APOS model.
alin = APOS(mdn=x.shape[1], mdo=y.shape[1], adn=0, mns=0)
_ = time.time()
alin.fit(x, y, steps=10)
alin_total = time.time() - _
print("alin_total: ", alin_total)

# Fit the linear model.
_ = time.time()
model, residuals = np.linalg.lstsq(x, y, rcond=None)[:2]
lin_total = time.time() - _
print(" lin_total: ", lin_total)
print(model)

lp = fmodpy.fimport("test_sgels.f90", lapack=True, omp=False).least_squares
xf = np.asarray(x.T, order="F")
yf = np.asarray(y, order="F")
_ = time.time()
s = lp(bytes('T','ASCII'), xf, yf)[-1]
lp_total = time.time() - _
print("  lp_total: ", lp_total)
print("s.shape: ", s.shape)
print(s)

# TODO:
#  - compute useless nodes in a layer by orthogonalization
#  - compute output values of noncontributors, least squares fit
#    the values that are kept to those noncontributing outputs
#  - update the weights of the remaining nodes and delete useless ones
#  - reduce the gradient afterwards to its principal components that
#    fill the space of values
#  - least squares fit the data at the layer before to the gradient
#    placed in the previously useless values
#  - orthogonalize all new values to see if rank has improved
#  - fill remaining empty rank with first K principal components of
#    preceeding layer
#  - pick shift terms for all new nodes as the value that maximizes
#    alignment with the gradient (first), and capturing all previous
#    layer values (principal components) second (shift = minval),
#    and increasing diversity of output (distance from neighbor
#    states) third
# 
