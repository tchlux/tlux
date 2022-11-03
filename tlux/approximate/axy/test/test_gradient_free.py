import os
from tlux.approximate.axy import AXY
import numpy as np

# Import codes that will be used for testing.
from tlux.plot import Plot
from tlux.random import well_spaced_box

seed = 0
np.random.seed(seed)

n = 2**8
d = 2
new_model = True
use_a = True
agg_dim = 64
agg_states = 4
use_x = False
model_dim = 64
model_states = 4
model_dim_output = 0
use_yi = False
steps = 1000
num_threads = None

ONLY_SGD = dict(
    faster_rate = 1.0,
    slower_rate = 1.0,
    update_ratio_step = 0.0,
    step_factor = 0.001,
    step_mean_change = 0.1,
    step_curv_change = 0.01,
    keep_best = True,
    basis_replacement = False,
)

settings = dict(
    seed=seed,
    early_stop = False,
    logging_step_frequency = 1,
    rank_check_frequency = 100,
    **({"mdo":model_dim_output} if model_dim_output is not None else {}),
    normalize_x = True,
    normalize_y = True,
    # **ONLY_SGD
)

# A function for testing approximation algorithms.
def f1(x):
    x = x.reshape((-1,2))
    x, y = x[:,0], x[:,1]
    return (3*x + np.cos(8*x)/2 + np.sin(5*y))

# A second function for testing approximation algorithms.
def f2(x):
    x = x.reshape((-1,2))
    return np.cos(np.linalg.norm(x,axis=1))

# A third function that is the trigonometric shift of the first.
def f3(x):
    x = x.reshape((-1,2))
    x, y = x[:,0], x[:,1]
    return (3*x + np.sin(8*x)/2 + np.cos(5*y))


functions = [f1, f2] # , f3]

# Generate data bounds.
x_min_max = [[-.1, 1.1], [-.1, 1.1]]

# Create all data.
points = [well_spaced_box(n, d) for _ in functions]
x =  np.concatenate(points, axis=0).astype("float32")
xi = np.asarray([n*[f.__name__] for f in functions], dtype=object).reshape((-1,1))
ax = x.reshape((-1,1)).copy()
axi = np.asarray(list(zip(
    [f.__name__ for f in functions for _ in range(n*d)],
    [i+1 for _ in range(n*len(functions)) for i in range(d)]
)), dtype=object)
sizes = np.ones(n*len(functions), dtype="int32") * d

# Concatenate the two different function outputs.
y = np.concatenate([f(p) for (f,p) in zip(functions, points)], axis=0).reshape((len(functions)*n,1))
# Generate classification data that is constructed by binning the existing y values.
yi = np.asarray([
    np.where(
        y[:,0] <= np.percentile(y[:,0], 50),
        'bottom',
        'top'
    ),
    np.where(
        y[:,0] <= np.percentile(y[:,0], 20),
        'small',
        np.where(
            y[:,0] <= np.percentile(y[:,0], 80),
            'medium',
            'large'
        )
    ),
], dtype=object).T
# Compute the numeric values associated with each Y group (mean of values in group).
yi_values = [{
    "bottom": y[y[:,0] <= np.percentile(y[:,0], 50)].mean(),
    "top": y[y[:,0] > np.percentile(y[:,0], 50)].mean(),
}, {
    "small": y[y[:,0] <= np.percentile(y[:,0], 20)].mean(),
    "medium": y[(y[:,0] > np.percentile(y[:,0], 20)) *
                (y[:,0] < np.percentile(y[:,0], 80))].mean(),
    "large": y[y[:,0] >= np.percentile(y[:,0], 80)].mean(),
}]

# Train a new model or load an existing one.
if (new_model or (not os.path.exists('temp-model.json'))):
    # Initialize a new model.
    print("Fitting model..")
    m = AXY(
        ads=agg_dim,
        ans=agg_states,
        mds=model_dim,
        mns=model_states,
        num_threads=num_threads,
        **settings,
    )
    m.fit(
        ax=(ax.copy() if use_a else None),
        axi=(axi if use_a else None),
        sizes=(sizes if use_a else None),
        x=(x.copy() if use_x else None),
        xi=(xi if use_x else None),
        y=y.copy(),
        yi=(yi if use_yi else None),
        steps=steps,
    )
    # Save and load the model.
    m.save("temp-model.json")

# Load the saved model.
m = AXY()
m.load("temp-model.json")
# Remove the saved model if only new models are desired.
if new_model:
    os.remove("temp-model.json")

# Print the model.
print()
print(m, m.config)
print()

def obj(params):
    m.model[:] = np.asarray(params, dtype="float32")
    # Evaluate the model at all training data.
    _ax = ax.copy()
    _axi = axi
    _sizes = sizes
    _x = x.copy()
    _xi = xi
    if (not use_a): _ax = _axi = _sizes = None
    if (not use_x): _x = _xi = None
    output = m.predict(ax=_ax, axi=_axi, sizes=_sizes, x=_x, xi=_xi)
    error = ((y - output)**2).sum()
    return float(error)

# --------------------------------------------------------
# Training by gradient free optimization.
# 
# The better this approach does, the worse the original trianing for the model performed.
# 
from util.optimize import minimize
sol = minimize(
    obj,
    solution=m.model.tolist(),
    bounds=[(-10,10)]*len(m.model),
    display=True,
    max_time=5,
)
m.model[:] = sol[:]
# --------------------------------------------------------

# Define a function that evaluates the model with some different presets.
def fhat(x, f=functions[0].__name__, yii=None):
    n = x.shape[0]
    xi = np.asarray([f]*n, dtype=object).reshape((-1,1))
    ax = x.reshape((-1,1)).copy()
    axi = np.asarray(list(zip(
        [f for _ in range(n*d)],
        [i+1 for _ in range(n) for i in range(d)]
    )), dtype=object)
    sizes = np.ones(x.shape[0], dtype="int32") * d
    if (not use_a): ax = axi = sizes = None
    if (not use_x): x = xi = None
    if (not use_yi): yii = None
    if (yii is None): return m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)[:,0]
    else:             return [yi_values[yii][v] for v in m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)[:,y.shape[1]+yii]]

# Generate the plot of the results.
print("Adding to plot..")
p = Plot()

# Add the provided points.
for i, (f, xf) in enumerate(zip(functions, points)):
    p.add(f"xi={f.__name__}", *xf.T, f(xf), color=i, group=i)

# Add the two functions that are being approximated.
for i,f in enumerate(functions):
    f = f.__name__
    p.add_func(f"xi={f}", lambda x: fhat(x, f=f), *x_min_max,
               group=i,  vectorized=True, color=i, opacity=0.8,
               plot_points=3000) # , mode="markers", marker_size=2)
if (use_yi):
    for f in functions:
        for i in range(yi.shape[1]):
            p.add_func(f"xi={f.__name__}, yi={i}", lambda x: fhat(x, f=f.__name__, yii=i),
                       *x_min_max, opacity=0.8, plot_points=3000,
                       mode="markers", shade=True, marker_size=4)

# Produce the visual.
print("Generating surface plot..")
# Generate a visual of the loss function.
if (len(getattr(globals().get("m",None), "record", [])) > 0):
    p.plot(show=False)
    print()
    print("Generating loss plot..")
    p = Plot("Mean squared error")
    # Rescale the columns of the record for visualization.
    record = m.record
    for i in range(0, record.shape[0]+1, max(1,record.shape[0] // 100)):
        step_indices = list(range(1,i+1))
        p.add("MSE", step_indices, record[:i,0], color=1, mode="lines", frame=i)
        p.add("Step factors", step_indices, record[:i,1], color=2, mode="lines", frame=i)
        p.add("Step sizes", step_indices, record[:i,2], color=3, mode="lines", frame=i)
        p.add("Update ratio", step_indices, record[:i,3], color=4, mode="lines", frame=i)
        p.add("Eval utilization", step_indices, record[:i,4], color=5, mode="lines", frame=i)
        p.add("Grad utilization", step_indices, record[:i,5], color=6, mode="lines", frame=i)
    p.show(append=True, show=True, y_range=[-.2, 1.2])
else:
    p.show()
print("", "done.", flush=True)
