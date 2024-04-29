import math
import time

import numpy as np
import tqdm

from tlux.profiling import profile
from tlux.approximate.balltree import BallTree 


r = 0.1
n = 10000
d = 512
t = "float32"
seed = 0
min_time_seconds = 5

print("Creating points..", flush=True)
np.random.seed(seed)
x = np.random.normal(
    size=(n,d),
).astype(t)



# print("Building tree..", flush=True)
# t = BallTree(x)
# print(t.in_radius(x[0], r))
# dists = np.linalg.norm(x - x[0], axis=1)
# print(np.arange(len(x))[dists <= r])
# exit()
# # 
# # 900 seconds per 100k pairwise
# def tree_pairwise(radius=0.1):
#     z = t.query(x[:74])


# 33 seconds per 100K pairwise
# 74 × 1M per second
@profile
def mmul_pairwise(min_sim=1.0 - r, max_dim=2**35):
    z = x.copy()
    z = (z.T / np.sqrt(np.sum(z**2, axis=1))).T
    block_size = min(z.shape[0], math.ceil(max_dim / (z.shape[0] * z.shape[1])))
    pairs = []
    for i in tqdm.tqdm(range(0, z.shape[0], block_size), delay=5, position=1, leave=False):
        all_pairs = z @ z[i:i+block_size].T
        for ji, jj in enumerate(range(i, min(z.shape[0], i+block_size))):
            above_min_sim = np.argwhere(all_pairs[jj+1:,ji] >= min_sim).flatten() + (jj+1)
            if above_min_sim.size > 0:
                pairs.append((jj, above_min_sim.tolist()))
    return pairs


# Measure the distance to all points, but then move to the next nearest point
#  and only measure distance to points that might be the radius.
@profile
def smart_mmul_pairwise(min_sim=1.0 - r, max_dim=2**35):
    pairs = []
    # Copy and normalize the data.
    z = x.copy()
    z = (z.T / np.sqrt(np.sum(z**2, axis=1))).T
    # Get the first item to do.
    to_do = np.arange(z.shape[0])
    i = to_do[0]
    to_do = to_do[1:]
    # Set up "to check" and "sims" for remaining computations.
    tc = np.arange(len(to_do))
    sims = np.zeros(len(to_do), dtype="float32")
    # Iterate until all pairs have been computed.
    # for _ in tqdm.tqdm(range(z.shape[0]-1), delay=5, position=1, leave=False):
    for _ in range(z.shape[0]-1):
        sims[tc] = z[to_do[tc]] @ z[i]
        # Compute point in min_sim.
        above_min_sim = np.argwhere(sims >= min_sim).flatten()
        if (above_min_sim.size > 0):
            pairs.append((i, to_do[above_min_sim].tolist()))
        # Pick next point to compare with others.
        nearest_i = np.argmax(sims)
        nearest_sim = sims[nearest_i]
        i = to_do[nearest_i]
        # Pop out the next element from "to_do" and "sims".
        to_do[nearest_i:-1] = to_do[nearest_i+1:]
        to_do = to_do[:-1]
        sims[nearest_i:-1] = sims[nearest_i+1:]
        sims = sims[:-1]
        # Figure out which points should be checked in the next round.
        sims[:] += (1.0 - nearest_sim)  # at most points got this much closer
        tc = (sims >= min_sim)
    pairs.sort()
    return pairs


# pairs = mmul_pairwise()
# print("pairs:", flush=True)
# for p in pairs:
#     print("", p)
# 
# print("", flush=True)
# pairs = smart_mmul_pairwise()
# print("pairs:", flush=True)
# for p in pairs:
#     print("", p)
# 
# exit()

@profile
def run_test():
    # tree_pairwise()
    mmul_pairwise()
    smart_mmul_pairwise()




print("Running tests..", flush=True)
start_sec = time.time()
for i in tqdm.tqdm(range(2**30), position=0, leave=True):
    run_test()
    if (time.time() - start_sec >= min_time_seconds):
        break


run_test.show_profile()
mmul_pairwise.show_profile()
smart_mmul_pairwise.show_profile()


# For radius queries we know:
#
# - D distance to root point
# - D+2 min distance to outer and inner child
# - 




# n = 1000
# ______________________________________________________________________
# File:     test/compare_with_matmul.py
# Function: run_test
# Memory:   33.59MB
# Time:     5.71s (5.68s accounted for)
# Line   Time         MDelta         Mem                 Execs  Code
# ----------------------------------------------------------------------
# call                               [43.94MB, 77.55MB]  10                        
# 31     0.00s (0%)   0.00MB (0%)    [43.94MB, 77.55MB]  10     @profile           
# 33     5.64s (99%)  11.95MB (36%)  [50.00MB, 77.59MB]  10         tree_pairwise()
# 34     0.04s (1%)   21.64MB (64%)  [59.86MB, 77.59MB]  10         mmul_pairwise()
# ----------------------------------------------------------------------


# n = 10000
# __________________________________________________________________________
# File:     test/compare_with_matmul.py
# Function: run_test
# Memory:   441.56MB
# Time:     55.50s (55.50s accounted for)
# Line   Time          MDelta          Mem                   Execs  Code
# --------------------------------------------------------------------------
# call                                 [95.41MB, 536.89MB]   10                        
# 30     0.00s (0%)    0.00MB (0%)     [95.41MB, 536.89MB]   10     @profile           
# 32     51.90s (94%)  28.80MB (7%)    [120.33MB, 536.97MB]  10         tree_pairwise()
# 33     3.60s (6%)    412.77MB (93%)  [532.75MB, 537.00MB]  10         mmul_pairwise()
# --------------------------------------------------------------------------
