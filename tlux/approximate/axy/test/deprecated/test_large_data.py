import numpy as np
from tlux.approximate.axy import AXY

v = 200000
n = 50000
an = 300
print("Allocating data..", flush=True)
axi = np.random.randint(0,v, (an*n,1)).astype("int32")
sizes = np.zeros(n, dtype="int32") + an
y = np.zeros(n, dtype="float32")
print("Building model..", flush=True)
m = AXY()
print("Fitting model..", flush=True)
m.fit(axi=axi, sizes=sizes, y=y, steps=10, num_threads=None)
# m.fit(axi=axi, sizes=sizes, y=y, steps=1, num_threads=1) # repeated call to see memory changes
print()
print("Evaluating model..")
e = m.predict(axi=axi, sizes=sizes, embeddings=True)
print(m)
print("Done.")


