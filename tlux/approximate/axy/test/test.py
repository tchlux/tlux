import numpy as np
from tlux.approximate.axy import AXY

print("_"*70)
print(" TESTING AXY MODULE")

# ----------------------------------------------------------------
#  Enable debugging option "-fcheck=bounds".
# import fmodpy
# fmodpy.config.f_compiler_args = "-fPIC -shared -O3 -fcheck=bounds"
# fmodpy.config.link_blas = ""
# fmodpy.config.link_lapack = ""
# fmodpy.config.link_blas = "-framework Accelerate"
# fmodpy.config.link_lapack = "-framework Accelerate"
# fmodpy.config.link_omp = ""
# ----------------------------------------------------------------

from tlux.plot import Plot
from tlux.random import well_spaced_ball, well_spaced_box

# TODO: test saving and loading with unique value maps
# TODO: design concise test function that has meaningful signal
#       in each of "ax", "axi", "x", "xi", test all combinations

# A function for testing approximation algorithms.
def f(x):
    x = x.reshape((-1,2))
    x, y = x[:,0], x[:,1]
    return (3*x + np.cos(8*x)/2 + np.sin(5*y))

n = 100
seed = 2
state_dim = 32
num_states = 8
steps = 1000
num_threads = None
np.random.seed(seed)
axy_kwargs = dict(
    num_threads = num_threads,
    seed = seed,
    logging_step_frequency = 1,
    rank_check_frequency = 1,
    min_update_ratio = 0.8,
    early_stop = False,
    # initial_shift_range=0.0,
    # faster_rate=1.0,
    # slower_rate=1.0,
    # basis_replacement = True,
    # orthogonalizing_step_frequency = 200,
    # keep_best = False,
    # step_replacement = 0.00,
    # equalize_y = True,
    # discontinuity=-1000.0
    # initial_step=0.01
)


TEST_FIT_SIZE = False
TEST_WEIGHTING = False
TEST_SAVE_LOAD = False
TEST_INT_INPUT = False
TEST_AGGREGATE = False
TEST_LARGE_MODEL = True
TEST_BAD_NUMBERS = False # Nan and Inf
TEST_SCALING = False # Very small (10^{-30}) and very large (10^{30}) values at once.
SHOW_VISUALS = True

if TEST_FIT_SIZE:
    # Aggreagtor model settings.
    dim_a_numeric = 4
    dim_a_embedding = None
    num_a_embeddings = 32
    dim_a_state = 8
    num_a_states = 2
    dim_a_out = None
    # Model settings.
    dim_m_numeric = 8
    dim_m_embedding = 32
    num_m_embeddings = 32
    dim_m_state = 64
    num_m_states = 8
    dim_m_out = 1
    # Number of points going into each model.
    na = 100000000
    nm = 100000
    # Initialize the model and its fit configuration.
    m = AXY(
        adn=dim_a_numeric, ade=dim_a_embedding, ane=num_a_embeddings,
        ads=dim_a_state, ans=num_a_states, ado=dim_a_out,
        mdn=dim_m_numeric, mde=dim_m_embedding, mne=num_m_embeddings,
        mds=dim_m_state, mns=num_m_states, mdo=dim_m_out,
        **axy_kwargs
    )
    print(m.config)    
    m.AXY.new_fit_config(nm, na, m.config)
    print()
    print(m)
    print(m.config)


if (not SHOW_VISUALS):
    class Plot:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, *args, **kwargs):
            return lambda *args, **kwargs: None


if TEST_SAVE_LOAD:
    # Try saving an untrained model.
    m = AXY()
    print("Empty model:")
    print("  str(model) =", str(m))
    print()
    m.save("testing_empty_save.json")
    m.load("testing_empty_save.json")
    from tlux.approximate import AXY
    m = AXY(mdn=2, mds=state_dim, mns=num_states, mdo=1, 
            steps=steps, **axy_kwargs,
    )
    print("Initialized model:")
    print(m)
    print()
    # Create the test plot.
    x = np.asarray(well_spaced_box(n, 2), dtype="float32", order="C")
    y = f(x).astype("float32")
    # Rescale the data to make it cover many orders of magnitude.
    if TEST_SCALING:
        x[:,0] *= 10.**30
        y[:] /= 10.**30
    # Add some invalid values to the data.
    if TEST_BAD_NUMBERS:
        x[len(x)//7 : 2*len(x)//7,0] = float('nan')
        x[-2:,1] = float('inf')
        y[len(y)//7 : 2*len(y)//7] = float('nan')
        y[-2:] = float('inf')
    # Construct weights.
    yw = np.random.random(size=n) ** 5
    yw = np.where(yw > 0.5, yw*2, 0.001)
    if (not TEST_WEIGHTING): yw[:] = 1.0
    # Fit the model.
    x_normalized = x.copy()
    y_normalized = y.copy()
    m.fit(x=x_normalized, y=y_normalized, yw=yw)
    # Do mean-substitution on the invalid numbers in the dataset.
    if TEST_BAD_NUMBERS:
        x = np.where(np.isfinite(x), x, np.nan)
        x[np.isnan(x)] = (np.nanmean(x, axis=0) * np.ones(x.shape))[np.isnan(x)]
        y = np.where(np.isfinite(y), y, np.nan)
        y[np.isnan(y)] = (np.nanmean(y, axis=0) * np.ones(y.shape))[np.isnan(y)]
    # Add the data and the surface of the model to the plot.
    p = Plot()
    x_min_max = np.asarray([x.min(axis=0), x.max(axis=0)]).T
    p.add("Data", *x[yw > 0.5].T, y[yw > 0.5])
    if (TEST_WEIGHTING):
        p.add("Data (low weight)", *x[yw <= 0.5].T, y[yw <= 0.5], color=3)
    # p.add("Normalized data", *x_fit.T, y_fit)
    p.add_func("Fit", m, *x_min_max, vectorized=True)
    # Show the normalized data.
    if TEST_BAD_NUMBERS:
        q = Plot()
        q.add("Normalized data", *x_normalized.T, y_normalized, color=4)
        q.show(append=True)
    # Try saving the trained model and applying it after loading.
    print("Saving model:")
    print(m)
    print()
    m.save("testing_real_save.json")
    m.load("testing_real_save.json")
    print("Loaded model:")
    print(m)
    print()
    my = m(x.copy())[:,0]
    my_lift = (my.max() - my.min()) * 0.025
    p.add("Loaded values", *x.T, my+my_lift, color=1, marker_size=4)
    p.plot(show=(m.record.size == 0))
    # Remove the save files.
    import os
    try: os.remove("testing_empty_save.json")
    except: pass
    try: os.remove("testing_real_save.json")
    except: pass


if TEST_INT_INPUT:
    print("Building model..")
    x = well_spaced_box(n, 2)
    x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
    y = f(x)
    # Initialize a new model.
    m = AXY(mdn=2, mds=state_dim, mns=num_states, mdo=1, mde=3, mne=2, steps=steps, **axy_kwargs)
    all_x = np.concatenate((x, x), axis=0)
    all_y = np.concatenate((y, np.cos(np.linalg.norm(x,axis=1))), axis=0)
    all_xi = np.concatenate((np.ones(len(x)),2*np.ones(len(x)))).reshape((-1,1)).astype("int32")
    x_fit = np.array(all_x, dtype="float32", order="C")
    m.fit(x=x_fit, xi=all_xi, y=all_y.copy())
    # Create an evaluation set that evaluates the model that was built over two differnt functions.
    xi1 = np.ones((len(x),1),dtype="int32")
    y1 = m(x, xi=xi1)
    y2 = m(x, xi=2*xi1)
    print("Adding to plot..")
    p = Plot()
    # p.add("x fit", *x_fit.T, all_y[:], color=0)
    p.add("xi=1 true", *x.T, all_y[:len(all_y)//2], color=0)
    p.add("xi=2 true", *x.T, all_y[len(all_y)//2:], color=1)
    p.add_func("xi=1", lambda x: m(x.copy(), xi=np.ones(len(x), dtype="int32").reshape((-1,1))), *x_min_max, vectorized=True, color=3, shade=True)
    p.add_func("xi=2", lambda x: m(x.copy(), xi=2*np.ones(len(x), dtype="int32").reshape((-1,1))), *x_min_max, vectorized=True, color=2, shade=True)
    # Generate the visual.
    print("Generating surface plot..")
    p.show(show=False)


if TEST_AGGREGATE:
    print("Building model..")
    x = well_spaced_box(n, 2)
    x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
    y = f(x)
    # Create all data.
    all_x = np.concatenate((x, x), axis=0)
    all_xi = np.concatenate((np.ones(len(x)),2*np.ones(len(x)))).reshape((-1,1)).astype("int32")
    ax = all_x.reshape((-1,1)).copy()
    axi = (np.ones(all_x.shape, dtype="int32") * (np.arange(all_x.shape[1])+1)).reshape(-1,1)
    sizes = np.ones(all_x.shape[0], dtype="int32") * 2
    all_y = np.concatenate((y, np.cos(np.linalg.norm(x,axis=1))), axis=0)
    all_y = all_y.reshape((all_y.shape[0],-1))
    # Initialize a new model.
    print("Fitting model..")
    m = AXY(
        mdn=0, adn=ax.shape[1], ado=2, mdo=all_y.shape[1], 
        ads=state_dim, ans=num_states, mds=state_dim, mns=num_states,
        ane=len(np.unique(axi.flatten())), mne=len(np.unique(all_xi.flatten())),
        **axy_kwargs,
    )
    m.fit(ax=ax.copy(), axi=axi, sizes=sizes, xi=all_xi, y=all_y.copy(), steps=steps)
    # Create an evaluation set that evaluates the model that was built over two differnt functions.
    xi1 = np.ones((len(x),1),dtype="int32")
    ax = x.reshape((-1,1)).copy()
    axi = (np.ones(x.shape, dtype="int32") * (np.arange(x.shape[1])+1)).reshape(-1,1)
    sizes = np.ones(x.shape[0], dtype="int32") * 2
    temp_x = np.zeros((x.shape[0],0), dtype="float32")
    y1 = m(x=temp_x, xi=xi1, ax=ax, axi=axi, sizes=sizes)
    y2 = m(x=temp_x, xi=2*xi1, ax=ax, axi=axi, sizes=sizes)
    print("Adding to plot..")
    p = Plot()
    p.add("xi=1 true", *x.T, all_y[:len(all_y)//2,0], color=0, group=0)
    p.add("xi=2 true", *x.T, all_y[len(all_y)//2:,0], color=1, group=1)
    def fhat(x, i=1):
        xi = i * np.ones((len(x),1),dtype="int32")
        ax = x.reshape((-1,1)).copy()
        axi = (np.ones(x.shape, dtype="int32") * (np.arange(x.shape[1])+1)).reshape(-1,1)
        sizes = np.ones(x.shape[0], dtype="int32") * 2
        temp_x = np.zeros((x.shape[0],0), dtype="float32")
        return m(x=temp_x, xi=xi, ax=ax, axi=axi, sizes=sizes)
    p.add_func("xi=1", lambda x: fhat(x, 1), [0,1], [0,1], vectorized=True, color=3, opacity=0.8, group=0) #, mode="markers", shade=True)
    p.add_func("xi=2", lambda x: fhat(x, 2), [0,1], [0,1], vectorized=True, color=2, opacity=0.8, group=1) #, mode="markers", shade=True)
    # Generate the visual.
    print("Generating surface plot..")
    p.show(show=False)


if (TEST_LARGE_MODEL):  # Takes ~40 seconds on 14" 2021 M1 Pro Macbook Pro, 32GB of RAM
    # Data size.
    nm = 100000
    na = 6000000
    adn = 10
    ane = 10
    mdn = 10
    mne = 10
    mdo = 10
    # Data to fit.
    ax = np.random.random(size=(na,adn)).astype("float32")
    axi = np.random.randint(0,ane,size=(na,1)).astype("float32")
    x = np.random.random(size=(nm,mdn)).astype("float32")
    xi = np.random.randint(0,mne,size=(nm,1)).astype("float32")
    y = np.random.random(size=(nm,mdo)).astype("float32")
    # Generate random starting indices for all the sizes and
    #   convert those starting indices into sizes of batches.
    sizes = np.arange(na, dtype="int32")
    np.random.shuffle(sizes)
    sizes = sizes[:nm]
    sizes[0] = 0
    sizes.sort()
    sizes[:-1] = sizes[1:] - sizes[:-1]
    sizes[-1] = na - sizes[-1]
    # Model settings.
    ans = 4
    ads = 64
    mns = 8
    mds = 64
    steps = 11
    kwargs = axy_kwargs.copy()
    kwargs.update(dict(
        rank_check_frequency = 0,
    ))
    # Fit model.
    m = AXY(adn=adn, ane=ane, mdn=mdn, mne=mne, mdo=mdo,
             ans=ans, ads=ads, mns=mns, mds=mds, **kwargs)
    print()
    print("Fitting large model..")
    m.fit(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, y=y,
          steps=steps, **kwargs)
    print()

# Generate a visual of the loss function.
if (SHOW_VISUALS and (len(getattr(globals().get("m",None), "record", [])) > 0)):
    print()
    print("Generating loss plot..")
    p = Plot("Mean squared error")
    # Rescale the columns of the record for visualization.
    record = m.record
    for i in range(0, record.shape[0], max(1,record.shape[0] // 100)):
        step_indices = list(range(i))
        p.add("MSE", step_indices, record[:i,0], color=1, mode="lines", frame=i)
        p.add("Step factors", step_indices, record[:i,1], color=2, mode="lines", frame=i)
        p.add("Step sizes", step_indices, record[:i,2], color=3, mode="lines", frame=i)
        p.add("Update ratio", step_indices, record[:i,3], color=4, mode="lines", frame=i)
        p.add("Eval utilization", step_indices, record[:i,4], color=5, mode="lines", frame=i)
        p.add("Grad utilization", step_indices, record[:i,5], color=6, mode="lines", frame=i)
    p.show(append=True, show=True, y_range=[-.2, 1.2])
print("", "done.", flush=True)

if ("m" in globals()):
    print()
    print(m)

