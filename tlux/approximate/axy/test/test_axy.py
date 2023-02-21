# TODO:
#  - Identify why (for large categorical aggregate inputs) the initial error would be very large.
#  - Verify the correctness of the gradient when AX and X are both provided.
#  - Create a function for visualizing all of the basis functions in a model.
#  - Make sure the above function works in higher dimension (use PCA?).
#  - 


from tlux.plot import Plot
import fmodpy
import numpy as np


# Get the directory for the AXY compiled source code.
AXY = fmodpy.fimport(
    input_fortran_file = "../axy.f90",
    dependencies = ["random.f90", "matrix_operations.f90", "sort_and_select.f90", "axy.f90"],
    name = "test_axy_module",
    blas = True,
    lapack = True,
    omp = True,
    wrap = True,
    # rebuild = True,
    verbose = False,
    f_compiler_args = "-fPIC -shared -O0 -pedantic -fcheck=bounds -ftrapv -ffpe-trap=invalid,overflow,underflow,zero",
).axy
# help(AXY)


# Most combinations of these define the possible uses of the model.
scenarios = dict(
    small_data = False,
    small_model = True,
    aggregate_batch_constrained = False,
    aggregate_numeric_input = True,
    aggregate_categorical_input = True,
    aggregate_model_layered = True,
    aggregator_only = False,
    fixed_batch_constrained = False,
    fixed_numeric_input = True,
    fixed_categorical_input = True,
    fixed_model_layered = True,
    numeric_output = True,
    categorical_output = True,
    weighted_output = False,
    weights_dimensioned = False,
    threaded = False,
)


# Given a scenario, generate a model and data to match the scenario.
def gen_config_data(scenario, seed=0):
    # Problem size.
    if scenario["small_data"]:
        nm_in = 10
        na_in = 100
        na_range = (0, 10)
        nm_range = (0, 10)
    else:
        nm_in = 10000
        na_in = 10000000
        na_range = (0, 10000)
        nm_range = (0, 10000)
    # Fixed batch constrained (cannot fit all fixed data in at once).
    if scenario["fixed_batch_constrained"]:
        nm = nm_in // 5
    else:
        nm = nm_in
    # Aggregate batch constrained (cannot fit all aggregate data in at once).
    if scenario["aggregate_batch_constrained"]:
        na = na_in // 5
    else:
        na = na_in
    # Aggregate numeric input.
    if scenario["aggregate_numeric_input"]:
        if scenario["small_data"]:
            adn = 2
        else:
            adn = 20
    else:
        adn = 0
    # Aggregate categorical input.
    if scenario["aggregate_categorical_input"]:
        if scenario["small_data"]:
            ane = 5
            adi = 1
        else:
            ane = 5000
            adi = 5
    else:
        ane = 0
        adi = 0
    # Aggregate model layered.
    if scenario["aggregate_model_layered"]:
        if scenario["small_model"]:
            ans = 2
            ads = 3
        else:
            ans = 20
            ads = 64
    else:
        ans = 0
        ads = 0
    # Fixed numeric input.
    if scenario["fixed_numeric_input"]:
        if scenario["small_data"]:
            mdn = 2
        else:
            mdn = 20
    else:
        mdn = 0
    # Fixed categorical input.
    if scenario["fixed_categorical_input"]:
        if scenario["small_data"]:
            mne = 5
            mdi = 1
        else:
            mne = 5000
            mdi = 5
    else:
        mne = 0
        mdi = 0
    # Fixed model layered.
    if scenario["fixed_model_layered"]:
        if scenario["small_model"]:
            mns = 2
            mds = 3
        else:
            mns = 20
            mds = 64
    else:
        mns = 0
        mds = 0
    # Aggregator only.
    if scenario["aggregator_only"]:
        mns = 0
        mds = 0
        mdo = 0
    # Numeric output.
    if scenario["numeric_output"]:
        if scenario["small_data"]:
            ydn = 2
        else:
            ydn = 20
    else:
        ydn = 0
    # Categorical output.
    if scenario["categorical_output"]:
        if scenario["small_data"]:
            yne = 5
            ydi = 1
        else:
            yne = 5000
            ydi = 5
    else:
        yne = 0
        ydi = 0
    # Weighted output dimensioned.
    if scenario["weighted_output"]:
        if scenario["weights_dimensioned"]:
            ywd = ydn + yne
        else:
            ywd = 1
    else:
        ywd = 0
    # Multithreading.
    if scenario["threaded"]:
        num_threads = None
    else:
        num_threads = 1
    # Generate the model config.
    config = AXY.new_model_config(
        adn=adn,
        ade=None,
        ane=ane,
        ads=ads,
        ans=ans,
        ado=None,
        mdn=mdn,
        mde=None,
        mne=mne,
        mds=mds,
        mns=mns,
        mdo=ydn + yne,
        num_threads=num_threads
    )
    # Generate a fit configuration.
    config = AXY.new_fit_config(
        nm=nm,
        na=na,
        nmt=nm_in,
        nat=na_in,
        adi=adi,
        mdi=mdi,
        seed=seed,
        config=config
    )
    # Seed the randomness for repeatability.
    np.random.seed(seed)
    # Generate data that matches these specifications.
    ftype = dict(order="F", dtype="float32")
    itype = dict(order="F", dtype="int32")
    asarray = lambda a, t: np.asarray(a, **t) if (a.size > 0) else None
    ax_in = asarray(np.random.random(size=(adn, na_in)), ftype)
    axi_in = asarray(np.random.randint(*na_range, size=(adi, na_in)), itype)
    sizes_in = asarray(np.random.randint(0, round(2*(na_in / nm_in)), size=(nm_in if na_in > 0 else 0,)), itype)
    x_in = asarray(np.random.random(size=(mdn, nm_in)), ftype)
    xi_in = asarray(np.random.randint(*nm_range, size=(mdi, nm_in)), itype)
    y_in = asarray(np.random.random(size=(ydn, nm_in)), ftype)
    yi_in = asarray(np.random.randint(0, yne-1, size=(ydi, nm_in)), itype)
    yw_in = asarray(np.random.random(size=(ywd, nm_in)), ftype)
    # Adjust the sizes to make sure the sum is the correct value.
    ssum = sum(sizes_in)
    i = 0
    while (ssum != na_in):
        if (ssum < na_in):
            ssum += 1
            sizes_in[i] += 1
        elif (ssum > na_in) and (sizes_in[i] > 0):
            ssum -= 1
            sizes_in[i] -= 1
        i = (i + 1) % len(sizes_in)
    # Data holders.
    ax = np.zeros((adn, na), **ftype)
    axi = np.zeros((adi, na), **itype)
    sizes = np.zeros(((nm if na_in > 0 else 0),), **itype)
    x = np.zeros((mdn, nm), **ftype)
    xi = np.zeros((mdi, nm), **itype)
    y = np.zeros((ydn, nm), **ftype)
    yi = np.zeros((ydi, nm), **itype)
    yw = np.zeros((ywd, nm), **ftype)
    # Return the config and data.
    return (
        config,
        dict(ax=ax_in, axi=axi_in, sizes=sizes_in, x=x_in, xi=xi_in, y=y_in, yi=yi_in, yw=yw_in),
        dict(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, y=y, yi=yi, yw=yw),
    )


# Define a scenario generator.
def scenario_generator(scenarios=scenarios, randomized=True, seed=None):
    # If this is randomized, overwrite the "range" function with a random one.
    if randomized:
        if (seed is not None):
            import random
            random.seed(0)
        from tlux.random import random_range as range
    else:
        import builtins
        range = builtins.range
    # Get all keys.
    keys = sorted(scenarios.keys())
    # Iterate over all binary pairs of keys.
    for i in range(2**len(keys)):
        scenario = scenarios.copy()
        bin_str = bin(i)[2:][::-1] + '0'*len(keys)
        for v,n in zip(bin_str, keys):
            scenario[n] = (v == '1')
        # Skip any configurations that affect the model when it is absent.
        if (scenario["aggregator_only"] and (
                scenario["fixed_numeric_input"] or
                scenario["fixed_categorical_input"] or
                scenario["fixed_model_layered"]
        )):
            pass
        # Skip the invalid configuration where outputs are NOT weighted and
        #  "weights_dimensioned" is True, that is meaningless since there are no weights.
        elif ((not scenario["weighted_output"]) and (scenario["weights_dimensioned"])):
            pass
        # Skip the invalid configuration where there are no outputs.
        elif (not (scenario["numeric_output"] or scenario["categorical_output"])):
            pass
        # Skip the invalid configuration for "large model" and NOT "model layered"
        elif not (scenario["small_model"] or 
                  scenario["fixed_model_layered"] or
                  scenario["aggregate_model_layered"]):
            pass
        # Skip scenarios involving the aggregator when it is not present.
        elif ((not scenario["aggregate_numeric_input"]) and
              (not scenario["aggregate_categorical_input"]) and
              (scenario["aggregate_batch_constrained"] or scenario["aggregate_model_layered"])):
            pass
        # Skip scenarios involving the model when it is not present.
        elif ((not scenario["fixed_numeric_input"]) and
              (not scenario["fixed_categorical_input"]) and
              (scenario["aggregate_batch_constrained"] or scenario["aggregate_model_layered"])):
            pass
        else:
            yield scenario


# TODO: Start modifying all the test routines to iterate over many scenarios.
seed = 0
for i, scenario in enumerate(scenario_generator()):
    config, data, work = gen_config_data(scenario, seed=seed)
    print()
    print(i)
    for n in sorted(scenario):
        print(f"  {str(scenario[n]):5s}  {n}")
    print(' data')
    for n in data:
        print(" ", n, data[n].shape if data[n] is not None else data[n])
    print(' temp')
    for n in work:
        print(" ", n, work[n].shape if work[n] is not None else work[n])
    if i == 8: exit()


# Test INIT_MODEL
def _test_init_model():
    seed = 0
    initial_shift_range = 1.0
    initial_output_scale = 0.1
    # Create a new config.
    config = AXY.new_model_config(
        adn = 2,
        ade = 3,
        ane = 204,
        ads = 3,
        ans = 20,
        ado = None,
        mdn = 1,
        mde = 3,
        mne = 1008,
        mds = 3,
        mns = 5,
        mdo = 1,
        num_threads = 30,
    )
    print()
    print(config)
    print()
    # Initialize the model.
    model = np.zeros(config.total_size, dtype="float32")
    AXY.init_model(config, model, seed=seed,
                   initial_shift_range=initial_shift_range,
                   initial_output_scale=initial_output_scale)
    # Store the model in a format that makes it easy to retrieve vectors.
    from tlux.approximate.axy import AxyModel
    m = AxyModel(config, model)
    print()
    print(m)
    print()
    p = Plot()
    p.add(f"{config.ane} Agg Embeddings", *m.a_embeddings, marker_size=3, marker_line_width=1)
    p.add(f"{config.mne} Mod Embeddings", *m.m_embeddings, marker_size=3, marker_line_width=1)
    for i in range(config.ans-1):
        p.add(f"Agg layer {i+1}", *m.a_state_vecs[:,:,i].T, marker_size=4, marker_line_width=1)
    for i in range(config.mns-1):
        p.add(f"Mod layer {i+1}", *m.m_state_vecs[:,:,i].T, marker_size=4, marker_line_width=1)
    p.show()


# _test_init_model()
# exit()


# Test COMPUTE_BATCHES
def _test_compute_batches():
    seed = 0
    np.random.seed(seed)
    initial_shift_range = 1.0
    initial_output_scale = 0.1
    # Create a new (null) config.
    config = AXY.new_model_config(
        adn = 0,
        mdn = 0,
        mdo = 0,
        num_threads = 30,
    )
    print()
    print(config)
    print()
    # Simple test with nice numbers.
    config.batch_size = 20
    # na = 100
    nm = 10
    # sizes = np.ones(nm, dtype="int32") * (na // nm)
    sizes = np.random.randint(0,10, size=(nm,)).astype("int32")
    na = sum(sizes)
    batcha_starts, batcha_ends, agg_starts, batchm_starts, batchm_ends, info = (
        AXY.compute_batches(config, na=na, nm=nm, sizes=sizes, joint=True, info=0)
    )
    print("nm: ", nm)
    print("sizes: ", sizes)
    print("sizes.sum: ", sum(sizes))
    print("batcha_starts.shape: ", batcha_starts.shape, batcha_starts.tolist())
    print("batcha_ends.shape:   ", batcha_ends.shape, batcha_ends.tolist())
    print("agg_starts.shape:    ", agg_starts.shape, agg_starts.tolist())
    print("batchm_starts.shape: ", batchm_starts.shape, batchm_starts.tolist())
    print("batchm_ends.shape:   ", batchm_ends.shape, batchm_ends.tolist())
    print("info: ", info)


_test_compute_batches()
exit()

# --------------------------------------------------------------------
#                           NORMALIZE_DATA

# Spawn a new model that is ready to be evaluated.
def spawn_model(adn, mdn, mdo, ade=0, ane=0, ads=3, ans=2, ado=None, mde=0, mne=0, mds=3, mns=2,
                seed=0, num_threads=30, initial_shift_range=1.0, initial_output_scale=0.1):
    # Create a new config.
    config = AXY.new_model_config(
        adn = adn,
        ade = ade,
        ane = ane,
        ads = ads,
        ans = ans,
        ado = ado,
        mdn = mdn,
        mde = mde,
        mne = mne,
        mds = mds,
        mns = mns,
        mdo = mdo,
        num_threads = num_threads,
    )
    # Initialize the model.
    model = np.zeros(config.total_size, dtype="float32")
    AXY.init_model(config, model, seed=seed,
                   initial_shift_range=initial_shift_range,
                   initial_output_scale=initial_output_scale)
    # Return the config and the model.
    return config, model


# Python implementation of an algorithm for clipping the sizes to fit.
def py_fix_sizes(na, nm, sizes):
    sorted_order = np.argsort(sizes[:nm])
    nremain = na
    for i in range(nm):
        current_total = sizes[sorted_order[i]] * (nm-i)
        if (current_total > nremain):
            max_agg = nremain // (nm-i)
            nextra = nremain - max_agg*(nm-i)
            sizes[sorted_order[i:]] = max_agg
            sizes[sorted_order[i:i+nextra]] += 1
            nremain = 0
            break
        nremain -= sizes[sorted_order[i]]
    return sizes[:nm], na - nremain

# Test FETCH_DATA (making sure it clips the NA sizes correctly).
def _test_fetch_data():
    nm_values = list(range(10, 101, 18))
    na_range = (1, 30)
    na_step = 20
    na_multitplier = 7
    for nm_in in nm_values:
        for na_max in range(0, 2*na_range[-1] + 1, na_step):
            na = na_max * nm_in
            for seed in range(6):
                # Set a seed.
                np.random.seed(seed)
                # Spawn a new model.
                config, model = spawn_model(adn=1, mdn=1, mdo=1, ade=0)
                # Create fake data that matches the shape of the problem.
                sizes_in = np.asarray(np.random.randint(*na_range, size=(nm_in,)), dtype="int32", order="F")
                na_in = sum(sizes_in)
                ax_in = np.asarray(np.random.random(size=(config.adn,na_in)), dtype="float32", order="F")
                axi_in = np.zeros((0,na_in), dtype="int32", order="F")
                x_in = np.asarray(np.random.random(size=(config.mdn,nm_in)), dtype="float32", order="F")
                xi_in = np.zeros((0,nm_in), dtype="int32", order="F")
                y_in = np.asarray(np.random.random(size=(config.mdo,nm_in)), dtype="float32", order="F")
                yw_in = np.asarray(np.random.random(size=(1,nm_in)), dtype="float32", order="F")
                # Create the containers for the data.
                if (nm_in > nm_values[len(nm_values)//2]):
                    nm = nm_in // 2
                else:
                    nm = nm_in
                ax = np.zeros((config.adn,na), dtype="float32", order="F")
                axi = np.zeros((0,na), dtype="int32", order="F")
                sizes = np.zeros((nm,), dtype="int32", order="F")
                x = np.zeros((config.mdn,nm), dtype="float32", order="F")
                xi = np.zeros((0,nm), dtype="int32", order="F")
                y = np.zeros((config.mdo,nm), dtype="float32", order="F")
                yw = np.zeros((1,nm), dtype="float32", order="F")
                # Run the size fixing code in python.
                py_sizes, py_na = py_fix_sizes(na, nm, sizes_in[:nm].copy())
                # Run the size fixing (and data fetching) code in Fortran.
                config = AXY.new_fit_config(nm=nm, nmt=nm_in, na=na, nat=na_in, seed=0, config=config)
                config.i_next = 0
                config.i_step = 1
                config.i_mult = 1
                config.i_mod = nm
                (
                    config, ax, axi, f_sizes, x, xi, y, yw, f_na
                ) = AXY.fetch_data(
                    config, model, ax_in, ax, axi_in, axi, sizes_in, sizes,
                    x_in, x, xi_in, xi, y_in, y, yw_in, yw
                )
                assert (py_na == f_na), f"Number of aggregate points did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_na}\n  fortran: {f_na}\n"
                assert (tuple(sorted(py_sizes.tolist())) == tuple(sorted(f_sizes.tolist()))), f"Sizes did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                assert max(abs(py_sizes-f_sizes)) in {0,1}, f"Sizes did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                if ((f_na >= na_in) and (len(sizes_in) == len(sizes))):
                    assert (tuple(ax[:,:na_in].tolist()) == tuple(ax_in[:,:].tolist())), f"AX did not match AX_IN even though there was enough space.\n  python:  {ax_in.tolist()}\n  fortran: {ax.tolist()}\n"

_test_fetch_data()


# --------------------------------------------------------------------
#                           NORMALIZE_DATA


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
def _test_normalize_data():
    # Generate data to test with.
    num_trials = 10
    size_range = (0, 10)
    np.random.seed(0)
    num_points = 100
    nm = 50
    na = 300
    dimension = 3
    from tlux.plot import Plot
    p = Plot()
    for i in range(num_trials):
        # Generate random data (that is skewed and placed off center).
        x, x_shift, x_scale, x_rotation = random_data(num_points, dimension)
        p.add('x '+str(i+1), *x.T, marker_size=3)
        y, y_shift, y_scale, y_rotation = random_data(num_points, dimension)
        p.add('y '+str(i+1), *y.T, marker_size=3)
        # Generate random AX data.
        sizes = np.asarray(np.random.randint(*size_range, size=(num_points,)), dtype="int32", order="F")
        num_aggregate = sizes.sum()
        ax, ax_shift, ax_scale, ax_rotation = random_data(num_aggregate, dimension)
        p.add('ax '+str(i+1), *ax.T, marker_size=3)
        # Set up all of the inputs to the NORMALIZE_DATA routine.
        config, model = spawn_model(adn=dimension, mdn=dimension, mdo=dimension, num_threads=1)
        config = AXY.new_fit_config(nm=nm, nmt=num_points, na=na, nat=num_aggregate, config=config)
        # Set up the "full data".
        ax_in = ax.T
        axi_in = np.zeros((0,num_aggregate), dtype="int32", order="F")
        sizes_in = sizes.copy()
        x_in = x.T
        xi_in = np.zeros((0,num_points), dtype="int32", order="F")
        y_in = y.T
        yw_in = np.zeros((0,num_points), dtype="float32", order="F")
        # Set up the "data holder".
        ax = np.zeros((config.adn,config.na), dtype="float32", order="F")
        axi = np.zeros((0,config.na), dtype="int32", order="F")
        sizes = np.zeros((config.nm if config.na > 0 else 0,), dtype="int32", order="F")
        x = np.zeros((config.mdi, config.nm), dtype="float32", order="F")
        xi = np.zeros((0,num_points), dtype="int32", order="F")
        y = np.zeros((config.do, config.nm), dtype="float32", order="F")
        yw = np.zeros((0,num_points), dtype="float32", order="F")
        # Set up the "transformations".
        ax_shift = np.zeros((config.adn,), dtype="float32", order="F")
        ax_rescale = np.zeros((config.adn,config.adn), dtype="float32", order="F")
        axi_shift = np.zeros((config.ade,), dtype="float32", order="F")
        axi_rescale = np.zeros((config.ade,config.ade), dtype="float32", order="F")
        ay_shift = np.zeros((config.ado,), dtype="float32", order="F")
        x_shift = np.zeros((config.mdn,), dtype="float32", order="F")
        x_rescale = np.zeros((config.mdn,config.mdn), dtype="float32", order="F")
        xi_shift = np.zeros((config.mde,), dtype="float32", order="F")
        xi_rescale = np.zeros((config.mde,config.mde), dtype="float32", order="F")
        y_shift = np.zeros((config.do,), dtype="float32", order="F")
        y_rescale = np.zeros((config.do,config.do), dtype="float32", order="F")
        # Set up the remaining data matrices.
        a_emb_vecs = model[config.asev-1:config.aeev].reshape((config.ane, config.ade)).T
        m_emb_vecs = model[config.msev-1:config.meev].reshape((config.mne, config.mde)).T
        a_out_vecs = model[config.asov-1:config.aeov].reshape((config.ado+1, config.adso)).T
        a_states = np.zeros((config.na, config.ads, config.ans+1), dtype="float32", order="F")
        ay = np.zeros((config.na, config.ado+1), dtype="float32", order="F")
        info = 0
        # Call the routine.
        config.rescale_y = True
        (
            config, model,
            ax_in, x_in, y_in, yw_in,
            ax, axi, sizes, x, xi, y, yw,
            ax_shift, ax_rescale, axi_shift, axi_rescale, ay_shift,
            x_shift, x_rescale, xi_shift, xi_rescale, y_shift, y_rescale,
            a_emb_vecs, m_emb_vecs,
            a_out_vecs, a_states, ay,
            info
        ) = AXY.normalize_data(
            config, model,
            ax_in, axi_in, sizes_in, x_in, xi_in, y_in, yw_in,
            ax, axi, sizes, x, xi, y, yw,
            ax_shift, ax_rescale, axi_shift, axi_rescale, ay_shift,
            x_shift, x_rescale, xi_shift, xi_rescale, y_shift, y_rescale,
            a_emb_vecs, m_emb_vecs, a_out_vecs, a_states, ay, info
        )
        p.add('-> x '+str(i+1), *x_in, marker_size=3)
        p.add('-> y '+str(i+1), *y_in, marker_size=3)
        p.add('-> ax '+str(i+1), *ax_in, marker_size=3)
        p.show()
        # TODO:
        #  - assert that the AX_IN, X_IN, and Y_IN are radialized
        #  - assert that the AY are radialized (ready for second model)
        #  - assert that the embedded AXI and XI are radialized
        #  - run this routine with inf and NaN inputs
        #  - run this routine with huge data (make sure memory usage doesn't explode)
        # 
        exit()

_test_normalize_data()




# 2023-02-18 10:48:07
# 
######################################################################################
# #     # Generate the radialized version.                                           #
# #     shift = np.zeros(dimension, dtype="float32")                                 #
# #     transform = np.zeros((dimension, dimension), dtype="float32", order="F")     #
# #     inverse = np.zeros((dimension, dimension), dtype="float32", order="F")       #
# #     to_flatten = (i % 2) == 0                                                    #
# #     descriptor = 'radialized' if to_flatten else 'normalized'                    #
# #     xr, shift, transform, inverse = mops.radialize(                              #
# #         x=(x.copy()).T, shift=shift, vecs=transform, inverse=inverse,            #
# #         flatten=to_flatten                                                       #
# #     )                                                                            #
# #     p.add(f"{i+1} {descriptor}", *xr, marker_size=3)                             #
#                                                                                    #
# #     # Use the inverse to "fix" the data back to its original form.               #
# #     xf = (inverse.T @ xr).T                                                      #
# #     xf -= shift                                                                  #
# #     p.add(f"{i+1} fixed", *xf.T, marker_size=3)                                  #
#                                                                                    #
# #     # Use the provided shift and transform to repeate the radialization process. #
# #     xrr = (x + shift) @ transform                                                #
# #     p.add(f"{i+1} re {descriptor}", *xrr.T, marker_size=3)                       #
# # p.show()                                                                         #
######################################################################################
