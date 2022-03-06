import fmodpy
lib = fmodpy.fimport("argselect.f90")

print(lib)

import numpy as np

print()
for i in range(100):
    n = 100
    a = np.random.random(size=(n,)).astype("float32")
    k = np.random.randint(1,n)
    i = lib.argselect(a, k) - 1
    v = a[i[k-1]]
    a.sort()
    t = a[k-1]
    assert (v == t), f"\nn: {n}\nk: {k}\ntruth: {t}\nobserved: {v}\na: {a}\ni: {i}"
    if (v != t):
        print(n, k, v == t)
