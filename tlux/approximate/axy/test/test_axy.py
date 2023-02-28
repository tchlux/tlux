# TODO:
#  - Identify why (for large categorical aggregate inputs) the initial error would be very large.
#  - Verify the correctness of the gradient when AX and X are both provided.
#  - Create a function for visualizing all of the basis functions in a model.
#  - Make sure the above function works in higher dimension (use PCA?).

from tlux.plot import Plot
from tlux.approximate.axy.test.scenarios import (
    scenarios,
    AXY,
    Details,
    check_code,
    spawn_model,
    gen_config_data,
    scenario_generator
)

import numpy as np
np.set_printoptions(linewidth=100)


# --------------------------------------------------------------------
#                        SCENARIO_GENERATOR


def _test_scenario_iteration(max_samples=8):
    # TODO: Start modifying all the test routines to iterate over many scenarios.
    seed = 0
    for i, scenario in enumerate(scenario_generator()):
        config, details, data, work = gen_config_data(scenario, seed=seed)
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
        if i == max_samples: exit()

# _test_scenario_iteration()
# exit()


# --------------------------------------------------------------------
#                           FIT_MODEL


def _test_large_data_fit():
    config, details, data, work = gen_config_data(dict(
        na_in=10000000,
        na=1000000,
        nm=10000
    ))
    model = details.model
    rwork = details.rwork
    iwork = details.iwork
    steps = details.steps
    record = details.record
    config.axi_normalized = False
    config.step_factor = 0.0001
    print()
    for n in sorted(scenarios):
        print(f"  {str(scenarios[n]):5s}  {n}")
    print(' data')
    for n in data:
        print(" ", n, data[n].shape if data[n] is not None else data[n])
    print(' temp')
    for n in work:
        print(" ", n, work[n].shape if work[n] is not None else work[n])
    print(' config')
    print(' ', config)
    (
        config, model, rwork, iwork, ax_in, x_in, y_in, yw_in,
        record, sum_squared_error, info
    ) = AXY.fit_model(
        config, model, rwork, iwork,
        data['ax_in'], data['axi_in'], data['sizes_in'],
        data['x_in'], data['xi_in'], data['y_in'], data['yw_in'],
        steps=steps, record=record
    )

    check_code(info, "fit_model")

    print("record: ")
    print(record.T)
    print()
    print("sum_squared_error: ")
    print(sum_squared_error)
    print()
    print("info: ", info)

# _test_large_data_fit()
# exit()


# --------------------------------------------------------------------
#                           INIT_MODEL


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


# --------------------------------------------------------------------
#                         COMPUTE_BATCHES


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


# _test_compute_batches()
# exit()


# --------------------------------------------------------------------
#                           FETCH_DATA

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


def _test_axi():
    # for s in scenario_generator():
    #     print(s)
    #     print()

    config, details, data, work = gen_config_data(dict(
        na_in=100000,
        na=1000000,
        adn=0,
        adi=1,
        ade=4,
        nm=10000,
        steps=10,
    ))
    print("config: ", config)
    for (k,v) in data.items():
        print("", k, (v.shape if hasattr(v, "shape") else v))

    seed = 0
    AXY.init_model(config, details.model, seed=seed)

    from tlux.math import svd
    from tlux.plot import Plot
    p = Plot()

    axi = details.a_embeddings.T
    vals, vecs = svd(axi)
    print("vals: ", vals)
    axi = axi @ (vecs[:3].T)
    p.add('axi before', *axi.T, marker_size=2)
    data.pop('yi_in', None)

    (
        config,
        model,
        rwork,
        iwork,
        ax_in,
        x_in, 
        y_in,
        yw_in,
        record,
        sse,
        info
    ) = AXY.fit_model(
        config=details.config,
        model=details.model,
        steps=details.steps,
        record=details.record,
        rwork=details.rwork,
        iwork=details.iwork,
        **data
    )

    axi = details.a_embeddings.T
    vals, vecs = svd(axi)
    print("vals: ", vals)
    axi = axi @ (vecs[:3].T)
    p.add('axi after', *axi.T, marker_size=2)

    p.show()

_test_axi()


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
    size_range = (0, 2000)
    np.random.seed(0)
    num_points = 1000
    nm = 500
    na = 300
    dimension = 64
    should_plot = (dimension <= 3) and (na + nm < 100000)
    if (should_plot):
        from tlux.plot import Plot
        p = Plot()
    for i in range(num_trials):
        # Generate random data (that is skewed and placed off center).
        x, x_shift, x_scale, x_rotation = random_data(num_points, dimension)
        if (should_plot):
            p.add('x '+str(i+1), *x.T, marker_size=3)
        y, y_shift, y_scale, y_rotation = random_data(num_points, dimension)
        if (should_plot):
            p.add('y '+str(i+1), *y.T, marker_size=3)
        # Generate random AX data.
        sizes = np.asarray(np.random.randint(*size_range, size=(num_points,)), dtype="int32", order="F")
        num_aggregate = sizes.sum()
        ax, ax_shift, ax_scale, ax_rotation = random_data(num_aggregate, dimension)
        if (should_plot):
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
        if (should_plot):
            p.add('-> x '+str(i+1), *x_in, marker_size=3)
            p.add('-> y '+str(i+1), *y_in, marker_size=3)
            p.add('-> ax '+str(i+1), *ax_in, marker_size=3)
            p.show()
        else:
            # TODO: Assert that the mean is near zero and the variance is near one.
            print()
            print("ax_in.shape:\n", ax_in.shape)
            print("ax_in.mean(axis=1):\n", ax_in.mean(axis=1))
            print("ax_in.std(axis=1):\n", ax_in.std(axis=1))
            print()
            print("y_in.mean(axis=1): \n", y_in.mean(axis=1))
            print("y_in.std(axis=1): \n", y_in.std(axis=1))
        #  - assert that the AX_IN, X_IN, and Y_IN are radialized
        #  - assert that the AY are radialized (ready for second model)
        #  - assert that the embedded AXI and XI are radialized
        #  - run this routine with inf and NaN inputs
        #  - run this routine with huge data (make sure memory usage doesn't explode)
        # 

# _test_normalize_data()
# exit()



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
