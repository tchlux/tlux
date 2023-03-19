import numpy as np
from tlux.random import well_spaced_box
from tlux.approximate.axy import AXY
from tlux.plot import Plot

# A function for testing approximation algorithms.
def f(x):
    x = x.reshape((-1,2))
    x, y = x[:,0], x[:,1]
    return (3*x + np.cos(8*x)/2 + np.sin(5*y))



x = well_spaced_box(100, 2)
y = f(x)

print("x: ", x.shape)
print("y.shape: ", y.shape)

m = AXY()
m.fit(x=x, y=y)

p = Plot()
p.add("data", *x.T, y)
p.add_func("mode", m, *([[-.1, 1.1]]*2), vectorized=True)
p.show()
