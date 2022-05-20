import time
import numpy as np
from tlux.approximate.balltree import BallTree
from sklearn.neighbors import KDTree
# from sklearn.neighbors import BallTree

print("Initializing data..")
np.random.seed(1)
x = np.random.random(size=(10000000, 16))
x[x.shape[0]//2:,:] = x[:x.shape[0]//2,:]
i = np.random.randint(x.shape[0])
print()

print("Building tree..")
time.sleep(0.1)
# s = time.time(); t = KDTree(x); e = time.time(); print(f"{e-s:.6f} second fit")
s = time.time(); t = BallTree(x); e = time.time(); print(f"{e-s:.6f} second fit")
print()

time.sleep(0.1)
s = time.time(); print(t.query(x[i:i+1])[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (1 nearest)")
print()

time.sleep(0.1)
s = time.time(); print(t.query(x[i:i+1], k=5)[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (5 nearest)")
print()

time.sleep(0.1)
s = time.time(); print(t.query(x[i:i+1], k=100)[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (100 nearest)")
print()
print()

try:
    time.sleep(0.1)
    s = time.time(); print(t.query(x[i:i+1], k=5, budget=100000)[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (100K budget)")
    print()

    time.sleep(0.1)
    s = time.time(); print(t.query(x[i:i+1], k=5, budget=100)[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (100 budget)")
    print()
    print()

    for i in range(10):
        time.sleep(0.01)
        s = time.time(); print(t.query(x[i:i+1], k=5, budget=10000, randomness=0.2)[0][0].tolist()); e = time.time(); print(f"{e-s:.6f} seconds (10K budget)")
        print()
except: pass

