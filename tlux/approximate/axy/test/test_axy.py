# TODO:
#  - Identify why (for large categorical aggregate inputs) the initial error would be very large.
#  - Verify the correctness of the gradient when AX and X are both provided.
#  - Create a function for visualizing all of the basis functions in a model.
#  - Make sure the above function works in higher dimension (use PCA?).

import fmodpy
# Get the directory for the AXY compiled source code.
AXY = fmodpy.fimport(
    input_fortran_file = "../axy.f90",
    dependencies = ["axy_random.f90", "axy_matrix_operations.f90", "axy_sort_and_select.f90", "axy.f90"],
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


from tlux.plot import Plot
from tlux.approximate.axy.summary import AxyModel
from tlux.approximate.axy.axy_random import random
from tlux.approximate.axy.test.scenarios import (
    SCENARIO,
    # AXY,
    Details,
    check_code,
    spawn_model,
    gen_config_data,
    scenario_generator,
    initialize_agg_iterator
)

import numpy as np
np.set_printoptions(linewidth=1000)


# --------------------------------------------------------------------
#                        SCENARIO_GENERATOR


def _test_scenario_iteration(max_samples=8):
    print("SCENARIO_GENERATOR")
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
    print(" passed")

# _test_scenario_iteration()
# exit()


# --------------------------------------------------------------------
#                           FIT_MODEL


def _test_large_data_fit():
    print("FIT_MODEL")
    config, details, data, work = gen_config_data(dict(
        na_in=10000000,
        na=1000000,
        nm=10000
    ))
    model = details.model
    rwork = details.rwork
    iwork = details.iwork
    lwork = details.lwork
    steps = details.steps
    record = details.record
    config.axi_normalized = False
    config.step_factor = 0.0001
    print()
    for n in sorted(SCENARIO):
        print(f"  {str(SCENARIO[n]):5s}  {n}")
    print(' data')
    for n in data:
        print(" ", n, data[n].shape if data[n] is not None else data[n])
    print(' temp')
    for n in work:
        print(" ", n, work[n].shape if work[n] is not None else work[n])
    print(' config')
    print(' ', config)
    (
        config, model, rwork, iwork, lwork, ax_in, x_in, y_in, yw_in,
        record, sum_squared_error, info
    ) = AXY.fit_model(
        config, model, rwork, iwork, lwork,
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
    print("INIT_MODEL")
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
    from tlux.approximate.axy.summary import AxyModel
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
    print("COMPUTE_BATCHES")
    # TODO:
    # 
    #  cases
    #   - more / less data than threads
    #   - joint / separate batching
    #   - batch size as small as 1 / as large as amount of data
    # 
    #  assertions
    #   - all data is covered
    #   - no overlap between batches
    #   - as many batches as possible = max(min(num_data, num_threads), num_data / batch_size)
    # 
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
    config.max_batch = 20
    # na = 100
    nm = 10
    # sizes = np.ones(nm, dtype="int32") * (na // nm)
    sizes = np.random.randint(0,10, size=(nm,)).astype("int64")
    na = sum(sizes)
    batcha_starts, batcha_ends, agg_starts, batchm_starts, batchm_ends, info = (
        AXY.compute_batches(config, na=na, nm=nm, sizes=sizes, joint=True, info=0)
    )

    # # Custom test.
    # na = 1
    # nm = 10
    # config.max_batch = 10000
    # config.num_threads = 2
    # sizes = np.asarray([1.0] + [0.0]*9, dtype="int64")
    # batcha_starts, batcha_ends, agg_starts, batchm_starts, batchm_ends, info = (
    #     AXY.compute_batches(config, na=na, nm=nm, sizes=sizes, joint=False, info=0)
    # )

    print("nm: ", nm)
    print("sizes: ", sizes)
    print("sizes.sum: ", sum(sizes))
    print("batcha_starts.shape: ", batcha_starts.shape, batcha_starts.tolist())
    print("batcha_ends.shape:   ", batcha_ends.shape, batcha_ends.tolist())
    print("agg_starts.shape:    ", agg_starts.shape, agg_starts.tolist())
    print("batchm_starts.shape: ", batchm_starts.shape, batchm_starts.tolist())
    print("batchm_ends.shape:   ", batchm_ends.shape, batchm_ends.tolist())
    print("batchm_starts: ", batchm_starts)
    print("batchm_ends: ", batchm_ends)
    print("batcha_starts: ", batcha_starts)
    print("batcha_starts: ", batcha_starts)
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
    print("FETCH_DATA")
    nm_values = list(range(10, 101, 18))
    na_range = (1, 30)
    na_step = 20
    na_multitplier = 7
    pairwise = False
    for nm_in in nm_values:
        pairwise = (not pairwise)
        for na_max in range(0, 2*na_range[-1] + 1, na_step):
            na = na_max * nm_in
            for seed in range(6):
                # Set a seed.
                np.random.seed(seed)
                # Spawn a new model.
                config, model = spawn_model(adn=1, mdn=1, mdo=1, ade=0)
                config.pairwise_aggregation = pairwise
                # Create fake data that matches the shape of the problem.
                sizes_in = np.asarray(np.random.randint(*na_range, size=(nm_in,)), dtype="int64", order="F")
                na_in = sum(sizes_in)
                ax_in = np.asarray(np.random.random(size=(config.adn,na_in)), dtype="float32", order="F")
                axi_in = np.zeros((0,na_in), dtype="int64", order="F")
                x_in = np.asarray(np.random.random(size=(config.mdn,nm_in)), dtype="float32", order="F")
                xi_in = np.zeros((0,nm_in), dtype="int64", order="F")
                y_in = np.asarray(np.random.random(size=(config.mdo,nm_in)), dtype="float32", order="F")
                yw_in = np.asarray(np.random.random(size=(1,nm_in)), dtype="float32", order="F")
                agg_iterators = np.zeros((5,nm_in), dtype="int64", order="F")
                initialize_agg_iterator(config, agg_iterators, sizes_in)
                for i in range(nm_in):
                    # Get next and do a full "circle" around the iterator, verifying its correctness.
                    seen = set()
                    *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i])
                    for _ in range(agg_iterators[0,i]):
                        seen.add(next_i)
                        *agg_iterators[1:,i], next_i = random.get_next_index(*agg_iterators[:,i])                        
                    seen = tuple( sorted(seen) )
                    expected = tuple( range(1,sizes_in[i]**(2 if pairwise else 1) + 1) )
                    assert (seen == expected), \
                        f"Aggregate iterator did not produce expected list of elements.\n  {seen}\n  {expected}\n  {agg_iterators[:,i]}"
                # Do batching for the second half of the tests.
                if (nm_in > nm_values[len(nm_values)//2]):
                    nm = nm_in // 2
                else:
                    nm = nm_in
                # Initialize work space.
                ax = np.zeros((config.adn,na), dtype="float32", order="F")
                axi = np.zeros((0,na), dtype="int64", order="F")
                sizes = np.zeros((nm,), dtype="int64", order="F")
                x = np.zeros((config.mdn,nm), dtype="float32", order="F")
                xi = np.zeros((0,nm), dtype="int64", order="F")
                y = np.zeros((config.mdo,nm), dtype="float32", order="F")
                yw = np.zeros((1,nm), dtype="float32", order="F")
                # Run the size fixing code in python.
                py_sizes, py_na = py_fix_sizes(na, nm, sizes_in[:nm]**(2 if pairwise else 1))
                # Run the size fixing (and data fetching) code in Fortran.
                config = AXY.new_fit_config(nm=nm, nmt=nm_in, na=na, nat=na_in, seed=0, config=config)
                config.i_next = 0
                config.i_step = 1
                config.i_mult = 1
                config.i_mod = nm
                # If all data can fit, make the aggregate iterators simple linear generators.
                if ((config.nm >= x_in.shape[1]) and (config.na >= ax_in.shape[1]) and (not pairwise)):
                    agg_iterators[1,:] = 0  # next
                    agg_iterators[2,:] = 1  # mult
                    agg_iterators[3,:] = 1  # step
                    agg_iterators[4,:] = agg_iterators[0,:]  # mod = limit
                # Call the Fortran library code.
                (
                    f_config, f_agg_iterators, f_ax, f_axi, f_sizes, f_x, f_xi, f_y, f_yw, f_na
                ) = AXY.fetch_data(
                    config, agg_iterators, ax_in, ax, axi_in, axi, sizes_in, sizes,
                    x_in, x, xi_in, xi, y_in, y, yw_in, yw
                )
                assert (py_na == f_na), f"Number of aggregate points did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_na}\n  fortran: {f_na}\n"
                assert (tuple(sorted(py_sizes.tolist())) == tuple(sorted(f_sizes.tolist()))), f"Sizes did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                assert max(abs(py_sizes-f_sizes)) in {0,1}, f"Sizes did not match. dict(nm = {nm}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                # Verify that when (nm >= nm_in) and (na >= na_in) that we get ALL inputs!
                if (config.nm >= x_in.shape[1]):
                    assert np.all(x_in == f_x), f"There was enough room for all x, but they were not there (in order).\n\n  x:   {x_in}\n  f_x: {f_x}"
                    assert np.all(xi_in == f_xi), f"There was enough room for all xi, but they were not there (in order).\n\n  xi:   {xi_in}\n  f_xi: {f_xi}"
                    assert np.all(y_in == f_y), f"There was enough room for all y, but they were not there (in order).\n\n  y:   {y_in}\n  f_y: {f_y}"
                    assert np.all(yw_in == f_yw), f"There was enough room for all yw, but they were not there (in order).\n\n  yw:   {yw_in}\n  f_yw: {f_yw}"
                    if ((config.na >= ax_in.shape[1]) and (not pairwise)):
                        assert np.all(sizes_in == f_sizes), f"There was enough room for all sizes, but they were not there (in order).\n\n  sizes:   {sizes_in}\n  f_sizes: {f_sizes}"
                        assert np.all(ax_in == f_ax[:,:ax_in.shape[1]]), f"There was enough room for all ax, but they were not there (in order).\n\n  ax:   {ax_in}\n  f_ax: {f_ax}"
                        assert np.all(axi_in == f_axi[:,:axi_in.shape[1]]), f"There was enough room for all axi, but they were not there (in order).\n\n  axi:   {axi_in}\n  f_axi: {f_axi}"


    print(" passed")

_test_fetch_data()


# --------------------------------------------------------------------
#                            EMBED
def _test_embed():
    print("EMBED")
    s = SCENARIO.copy()
    s.update(dict(
        pairwise_aggregation = True,
        input_aggregate_categorical = True,
        input_aggregate_numeric = True,
        input_fixed_categorical = False,
        input_fixed_numeric = False,
        batch_aggregate_constrained = False,
        batch_fixed_constrained = False,
        model_aggregate_layered = True,
        model_fixed_layered = True,
        small_data = True,
        small_model = True,
        adi = 2,
        na_in = 2,
        na    = 2,
        nm_in = 1,
        nm    = 1,
    ))
    config, details, raw_data, data = gen_config_data(scenario=s, seed=0)
    config.pairwise_aggregation = True
    config.na = sum(raw_data["sizes_in"]**2)
    config.i_next = 0
    config.i_step = 1
    config.i_mult = 1
    config.i_mod = s["nm"]
    yi_in = raw_data.pop("yi_in")
    yi = data.pop("yi")
    axi_in = raw_data["axi_in"]
    sizes_in = raw_data["sizes_in"]
    axi_in[0,:] = 0
    details.agg_iterators[0,:] = sizes_in[:]**2 # limit
    details.agg_iterators[1,:] = 0 # next
    details.agg_iterators[2,:] = 1 # mult
    details.agg_iterators[3,:] = 1 # step
    details.agg_iterators[4,:] = sizes_in[:]**2 # mod
    print("axi_in: \n", axi_in.T)
    (
        config, agg_iterators, ax, axi, f_sizes, x, xi, y, yw, f_na
    ) = AXY.fetch_data(
        config=config, agg_iterators=details.agg_iterators, **raw_data, **data,
    )
    print("axi: \n", axi.T)
    # Once we have fetched data that includes pairs, we should verify that they are embedded correctly.
    # 
    # TODO: Verify that the pairs produced by FETCH_DATA match the pairs retrieved by EMBED and EMBEDDING_GRADIENT

# _test_embed()
# exit()

# --------------------------------------------------------------------
#                           EVALUATE

# Define a function that correct casts the data to the desired type.
def cast(arr, dtype):
    arr = np.asarray(arr)
    if (type(dtype) is type):
        arr = np.asarray(
            [dtype(v) for v in arr.flatten()],
            dtype=dtype,
            order="F"
        ).reshape(arr.shape)            
    else:
        arr = np.asarray(arr, dtype=dtype, order="F")
    return arr

# Define the full EVALUATE function in python (testing via reimplementation).
def py_evaluate(config, model, ax, axi, sizes, x, xi, dtype="float32", **unused_kwargs):
    # Get some constants.
    nm = x.shape[1]
    na = ax.shape[1]
    m = AxyModel(config, cast(model, dtype))
    # Embed the AXI values.
    ax_embedded = cast(np.zeros((config.adi, na)), dtype)
    for n in range(axi.shape[1]):
        ax_embedded[:config.adn,n] = ax[:config.adn,n]
        for d in range(axi.shape[0]):
            e = axi[d,n]
            if (e > 0) and (e <= config.ane):
                ax_embedded[-config.ade:,n] += m.a_embeddings[:,e-1]
            elif (e > config.ane):
                e1, e2 = random.index_to_pair(max_value=config.ane+1, i=e-config.ane)
                ax_embedded[-config.ade:,n] += m.a_embeddings[:,e1-1-1] - m.a_embeddings[:,e2-1-1]
        if (axi.shape[0] > 1):
            ax_embedded[-config.ade:,n] /= axi.shape[0]
    ax = ax_embedded
    # Embed the XI values.
    x_embedded = cast(np.zeros((config.mdi, nm)), dtype)
    for n in range(xi.shape[1]):
        x_embedded[:config.mdn,n] = x[:config.mdn,n]
        for d in range(xi.shape[0]):
            e = xi[d,n]
            if (e > 0) and (e <= config.mne):
                x_embedded[config.mdn:config.mdn+config.mde:,n] += m.m_embeddings[:,e-1]
            elif (e > config.mne):
                e1, e2 = random.index_to_pair(max_value=config.mne, i=e)
                x_embedded[config.mdn:config.mdn+config.mde,n] += m.m_embeddings[:,e1-1] - m.m_embeddings[:,e2-1]
        if (xi.shape[0] > 1):
            x_embedded[-config.mde:,n] /= xi.shape[0]
    x = x_embedded
    # Initialize a holder for the output.
    y = cast(np.zeros((config.mdo, nm)), dtype)
    # Evaluate the aggregator.
    if (config.ado > 0):
        if (config.normalize and (config.adn > 0)):
            # Add the shift term (across all point column vectors).
            ax[:config.adn,:] = (ax[:config.adn,:].T + m.ax_shift).T
            # Replace all NaN or Inf values
            ax[:config.adn,:] = np.where(
                np.logical_or(
                    np.isnan(ax[:config.adn,:].astype("float32")),
                    np.isinf(ax[:config.adn,:].astype("float32")),
                ),
                0.0,
                ax[:config.adn,:]
            )
            # Apply the input multiplier.
            if (config.needs_scaling):
                ax[:config.adn,:] = m.ax_rescale.T @ ax[:config.adn,:]
        # Evaluate the MLP.
        values = ax
        if (config.ans > 0):
            # Apply input transformation.
            values = np.clip(
                ((values.T @ m.a_input_vecs) + m.a_input_shift).T,
                config.discontinuity, float('inf')
            )
            # Apply internal transformations.
            for i in range(config.ans-1):
                values = np.clip(
                    ((values.T @ m.a_state_vecs[:,:,i]) + m.a_state_shift[:,i]).T,
                    config.discontinuity, float('inf')
                )
        # Apply output transformation.
        ay = (values.T @ m.a_output_vecs[:,:])
        ay_error = ay[:,config.ado:config.ado+1]  # extract +1 for error prediction
        ay = ay[:,:config.ado] # strip off +1 for error prediction
        # If there is a following model..
        if (config.mdo > 0):
            # Apply output shift.
            ay[:,:] = (ay + m.ay_shift)
            # Compute the first aggregator output embedding position.
            e = config.mdn + config.mde
            # Aggregate the batches.
            a = 0
            for i,s in enumerate(sizes):
                if (s > 0):
                    x[e:,i] = ay[a:a+s,:config.ado].sum(axis=0) / s
                else:
                    x[e:,i] = 0
                a += s
        else:
            a = 0
            for i,s in enumerate(sizes):
                if (s > 0):
                    y[:,i] = ay[a:a+s,:config.ado].sum(axis=0) / s
                else:
                    y[:,i] = 0
                a += s
    # Evaluate the fixed model.
    if (config.mdo > 0):
        if (config.normalize and (config.mdn > 0)):
            # Add the shift term (across all point column vectors).
            x[:config.mdn,:] = (x[:config.mdn,:].T + m.x_shift).T
            # Replace all NaN or Inf values
            x[:config.mdn,:] = np.where(
                np.logical_or(
                    np.isnan(x[:config.mdn,:].astype("float32")),
                    np.isinf(x[:config.mdn,:].astype("float32")),
                ),
                0.0,
                x[:config.mdn,:]
            )
            # Apply the input multiplier.
            if (config.needs_scaling):
                x[:config.mdn,:] = m.x_rescale.T @ x[:config.mdn,:]
        # Evaluate the MLP.
        values = x
        if (config.mns > 0):
            # Apply input transformation.
            values = np.clip(
                ((values.T @ m.m_input_vecs) + m.m_input_shift).T,
                config.discontinuity, float('inf')
            )
            # Apply internal transformations.
            for i in range(config.mns-1):
                values = np.clip(
                    ((values.T @ m.m_state_vecs[:,:,i]) + m.m_state_shift[:,i]).T,
                    config.discontinuity, float('inf')
                )
        # Apply output transformation.
        y = (values.T @ m.m_output_vecs[:,:]).T
    # Apply final normalization.
    if (config.normalize):
        if (config.needs_scaling):
            y[:,:] = m.y_rescale.T @ y
        y[:,:] = (y.T + m.y_shift).T
    # Return the final values.
    return y


def _test_evaluate():
    print("EVALUATE")
    # Generate a scenario.
    s = SCENARIO.copy()
    s["input_aggregate_categorical"] = True
    s["input_aggregate_numeric"] = True
    s["input_fixed_categorical"] = True
    s["input_fixed_numeric"] = True
    s["batch_aggregate_constrained"] = False
    s["batch_fixed_constrained"] = False
    s["model_aggregate_layered"] = True,
    s["model_fixed_layered"] = True,
    s["small_data"] = True
    s["small_model"] = False
    s["normalize"] = False
    s["needs_scaling"] = False
    s["num_threads"] = 10
    s["na_in"] = 100000
    s["nm_in"] = 10000
    s["na"] = s["na_in"]
    s["nm"] = s["nm_in"]
    config, details, raw_data, data = gen_config_data(scenario=s, seed=0)
    config.pairwise_aggregation = True
    model = details.model

    summary = ""
    summary += f"AxyModel(config, model):  {AxyModel(config, model, show_times=False)}\n"
    summary += "Input data:\n"
    for (k,v) in raw_data.items():
        summary += f" {v.dtype if hasattr(v, 'dtype') else type(v).__name__} {k} {v.shape if hasattr(v, 'shape') else ''}\n"
    summary += "\n"
    summary += "Batch data:\n"
    for (k,v) in data.items():
        summary += f" {v.dtype if hasattr(v, 'dtype') else type(v).__name__} {k} {v.shape if hasattr(v, 'shape') else ''}\n"
    summary += "\n"

    # Fetch and data (for model evaluation).
    initialize_agg_iterator(config, details.agg_iterators, raw_data["sizes_in"])
    (
        config, details.agg_iterators,
        data["ax"], data["axi"], data["sizes"],
        data["x"], data["xi"], data["y"], data["yw"],
        data["na"]
    ) = AXY.fetch_data(
        config=config, agg_iterators=details.agg_iterators,
        ax_in=raw_data["ax_in"], ax=data["ax"],
        axi_in=raw_data["axi_in"], axi=data["axi"],
        sizes_in=raw_data["sizes_in"], sizes=data["sizes"],
        x_in=raw_data["x_in"], x=data["x"],
        xi_in=raw_data["xi_in"], xi=data["xi"],
        y_in=raw_data["y_in"], y=data["y"],
        yw_in=raw_data["yw_in"], yw=data["yw"],        
    )

    # All necessary keyword arguments for model evaluation.
    na = data["na"]
    eval_kwargs = dict(axi=data["axi"][:,:na], ax=data["ax"][:,:na],
                       ay=details.ay[:na,:], sizes=data["sizes"],
                       x=data["x"], xi=data["xi"], y=data["y"],
                       a_states=details.a_states[:na,:,:], m_states=details.m_states)

    # Fortran evaluation.
    def f_evaluate(config, model, **data):
        local_data = {k:v.copy(order="F") for (k,v) in data.items()}
        local_data["y"] *= 0.0
        AXY.embed(
            config, model,
            axi=local_data["axi"], xi=local_data["xi"],
            ax=local_data["ax"], x=local_data["x"]
        )
        local_data.pop("axi")
        local_data.pop("xi")
        local_info = 0
        (
            config,
            ax, ay, x,
            y, a_states, m_states,
            local_info
        ) = AXY.evaluate(
            config, model, info=local_info,
            **local_data
        )
        check_code(local_info, "AXY.evaluate")
        return y

    fy = f_evaluate(config, model, **eval_kwargs)
    pyy = py_evaluate(config, model, **eval_kwargs)
    maxdiff = np.max(np.abs(fy - pyy))
    assert maxdiff < (2**(-13)), f"ERROR: Failed comparison between Fortran and Python implementations of AXY.evaluate.\n  Max difference observed was {maxdiff}.\n  Summary of situation:\n\n{summary}"
    print(" passed")

_test_evaluate()


# --------------------------------------------------------------------
#                           MODEL_GRADIENT
def _test_model_gradient():
    print("MODEL_GRADIENT")

    show_results = False

    # Generate a scenario.
    s = SCENARIO.copy()
    s["input_aggregate_categorical"] = True
    s["input_aggregate_numeric"] = True
    s["input_fixed_categorical"] = True
    s["input_fixed_numeric"] = True
    s["batch_aggregate_constrained"] = False
    s["batch_fixed_constrained"] = False
    s["model_aggregate_layered"] = False
    s["model_fixed_layered"] = False
    s["small_data"] = True
    s["small_model"] = True
    s["num_threads"] = 1
    s["pairwise_aggregation"] = True
    # s["adn"] = 1
    # s["ade"] = 4
    # s["ans"] = 1
    # s["ads"] = 8
    # s["ado"] = 4
    # s["mde"] = 0
    # s["mdn"] = 0
    # s["mns"] = 1
    # s["mds"] = 8
    s["na_in"] = 20
    s["nm_in"] = 10
    s["na"] = s["na_in"]
    s["nm"] = s["nm_in"]
    config, details, raw_data, data = gen_config_data(
        scenario=s,
        seed=0,
        # ane=3,
    )
    config.pairwise_aggregation = True
    model = details.model
    details.ay_shift *= 0.0
    # Fetch and embed data (for model evaluation).
    initialize_agg_iterator(config, details.agg_iterators, raw_data["sizes_in"])
    (
        config, details.agg_iterators,
        data["ax"], data["axi"], data["sizes"],
        data["x"], data["xi"], data["y"], data["yw"],
        data["na"]
    ) = AXY.fetch_data(
        config=config, agg_iterators=details.agg_iterators,
        ax_in=raw_data["ax_in"], ax=data["ax"],
        axi_in=raw_data["axi_in"], axi=data["axi"],
        sizes_in=raw_data["sizes_in"], sizes=data["sizes"],
        x_in=raw_data["x_in"], x=data["x"],
        xi_in=raw_data["xi_in"], xi=data["xi"],
        y_in=raw_data["y_in"], y=data["y"],
        yw_in=raw_data["yw_in"], yw=data["yw"],        
    )
    na = data["na"]
    eval_kwargs = dict(axi=data["axi"][:,:na], ax=data["ax"][:,:na],
                       ay=details.ay[:na,:], sizes=data["sizes"],
                       x=data["x"], xi=data["xi"], y=data["y"],
                       a_states=details.a_states[:na,:,:], m_states=details.m_states)

    # Approximate the gradient with a finite difference.
    from tlux.math.fraction import Fraction
    dtype = Fraction
    # dtype = "float64"
    offset = Fraction(1, 2**52)
    def finite_difference_gradient(model, data, config=config, offset=offset, dtype=dtype):
        model = cast(model, dtype)
        y = cast(data["y"], dtype)
        n = y.shape[1]
        # Evaluate the model and compute the current error.
        fy = py_evaluate(config, model, dtype=dtype, **data)
        squared_error = (fy - y)**2 / 2
        # Initialize the gradient to zero.
        gradient = 0 * model.copy()[:config.num_vars]
        # For each component of the model, modify the model, evaluate, observe change in error.
        for i in range(config.num_vars):
            local_model = model.copy()
            local_model[i] += offset
            local_fy = py_evaluate(config, local_model, dtype=dtype, **data)
            local_squared_error = (local_fy - y)**2 / 2
            error_delta = (local_squared_error - squared_error).sum() / n
            gradient[i] = error_delta / offset
        # Show the "correct" gradient calculation.
        _ = AxyModel(config, model)
        ay_grad = _.m_output_vecs @ (fy-y)
        return np.concatenate((gradient, model[config.num_vars:]))

    # Compute the gradient with the library function.
    def axy_gradient(model, data, config=config, details=details):
        # Define the copy of the data that will be used for evaluating the model.
        info = 0
        # The "na" key should contain the actual number of aggregate values (after fetch).
        na = data["na"]
        axi = data["axi"][:,:na].copy(order="F")
        xi = data["xi"].copy(order="F")
        ax = data["ax"][:,:na].copy(order="F")
        x = data["x"].copy(order="F")
        y = data["y"].copy(order="F")
        yw = data["yw"].copy(order="F")
        sizes = data["sizes"].copy(order="F")
        ay = details.ay[:na,:].copy(order="F")
        a_states = details.a_grads[:na,:,:].copy(order="F")
        m_states = details.m_grads.copy(order="F")
        fy = data["y"].copy(order="F")
        # Embed the data.
        (
            config,
            ax,
            x
        ) = AXY.embed(
            config, model,
            axi,
            xi,
            ax,
            x
        )
        # Evaluate the model (to populate intermediate state value holders).
        (
            config, ax, ay_gradient, x, y_gradient, a_grads, m_grads, info
        ) = AXY.evaluate(
            config, model,
            ax=ax,
            ay=ay,
            sizes=sizes,
            x=x,
            y=fy,
            a_states=a_states,
            m_states=m_states,
            info=0
        )
        check_code(info, "AXY.evaluate")

        model_grad = 0 * details.model_grad
        a_emb_temp = 0 * details.a_emb_temp
        m_emb_temp = 0 * details.m_emb_temp
        (
            config,
            ax,
            x,
            sum_squared_gradient,
            model_grad,
            info,
            ay_gradient,
            y_gradient,
            a_grads,
            m_grads,
            a_emb_temp,
            m_emb_temp,
        ) = AXY.model_gradient(
            config, model,
            ax=ax, axi=axi, sizes=sizes,
            x=x, xi=xi,
            y=y, yw=yw,
            sum_squared_gradient=0.0,
            model_grad=model_grad,
            info=info,
            ay_gradient=ay_gradient,
            y_gradient=y_gradient,
            a_grads=a_grads,
            m_grads=m_grads,
            a_emb_temp=a_emb_temp,
            m_emb_temp=m_emb_temp,
        )
        check_code(info, "AXY.model_gradient")
        return np.concatenate((model_grad.sum(axis=1), model[config.num_vars:]))

    gradient = finite_difference_gradient(model, data)
    model_gradient = axy_gradient(model, data)
    error = (model_gradient - gradient)
    ratio = np.asarray([1 + guess if (val == 0) else guess / val
                        for (guess,val) in zip(model_gradient, gradient)])
    max_error = abs(error).max()
    max_ratio_error = abs(ratio-1).max()
    failed_test = (max_error > 0.0001) and (max_ratio_error > 0.01)

    if (failed_test or show_results):
        print("AxyModel(config, model): ")
        print(AxyModel(config, model, show_vecs=True, show_times=False))
        print("error: ", error[np.argsort(-abs(error))[:5]].astype("float32"))
        print("ratio: ", ratio[np.argsort(abs(ratio-1))[:5]].astype("float32"))
        print("max_error:       ", float(max_error))
        print("max_ratio_error: ", float(max_ratio_error))
        print()
        # Show the configuration and the data.
        print()
        print('-'*100)
        print("config: ", config)
        print()
        print("Data:")
        for (k,v) in data.items():
            print("", type(k).__name__, k, v.shape if hasattr(v,"shape") else "")
        print('-'*100)
        print()

        # Show the model.
        print("-"*70)
        print("        MODEL")
        print()
        print(AxyModel(config, model, show_vecs=True, show_times=False))

        # Get the gradient via finite difference.
        print("-"*70)
        print("        TRUTH")
        print()
        print(AxyModel(config, gradient.astype("float64"), show_vecs=True, show_times=False))

        # Get the gradient with the compiled library.
        print("-"*70)
        print("        AXY")
        print()
        print(AxyModel(config, model_gradient, show_vecs=True, show_times=False))

        # Show the difference between the provided gradient and the "correct" (approximately) gradient.
        print("-"*70)
        print("        ERROR")
        print()
        print(AxyModel(config, error.astype("float64").round(4), show_vecs=True, show_times=False))

        # Show the ratio between the "correct" gradient and the one provided by the model.
        print("-"*70)
        print("        RATIO")
        print()
        print(AxyModel(config, ratio.astype("float64").round(2), show_vecs=True, show_times=False))
        print()
        
        assert (not failed_test), f"Either the maximum error ({float(max_error)}) was too high or the ratio between the exact gradient and computed gradient ({float(max_ratio_error)}) was too far from 1."
    else:
        print(" passed")

_test_model_gradient()


# --------------------------------------------------------------------
#                              AXI

def _test_axi():
    print("FIT_MODEL")
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
        lwork,
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
        lwork=details.lwork,
        **data
    )

    axi = details.a_embeddings.T
    vals, vecs = svd(axi)
    print("vals: ", vals)
    axi = axi @ (vecs[:3].T)
    p.add('axi after', *axi.T, marker_size=2)

    p.show()

# _test_axi()
# exit()


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
    print("NORMALIZE_DATA")
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
        sizes = np.asarray(np.random.randint(*size_range, size=(num_points,)), dtype="int64", order="F")
        num_aggregate = sizes.sum()
        ax, ax_shift, ax_scale, ax_rotation = random_data(num_aggregate, dimension)
        if (should_plot):
            p.add('ax '+str(i+1), *ax.T, marker_size=3)
        # Set up all of the inputs to the NORMALIZE_DATA routine.
        config, model = spawn_model(adn=dimension, mdn=dimension, mdo=dimension, num_threads=1)
        config.pairwise_aggregation = True
        config = AXY.new_fit_config(nm=nm, nmt=num_points, na=na, nat=num_aggregate, config=config)
        # Set up the "full data".
        ax_in = ax.T
        axi_in = np.zeros((0,num_aggregate), dtype="int64", order="F")
        sizes_in = sizes.copy()
        x_in = x.T
        xi_in = np.zeros((0,num_points), dtype="int64", order="F")
        y_in = y.T
        yw_in = np.zeros((0,num_points), dtype="float32", order="F")
        # Set up the "data holder".
        ax = np.zeros((config.adn,config.na), dtype="float32", order="F")
        axi = np.zeros((0,config.na), dtype="int64", order="F")
        sizes = np.zeros((config.nm if config.na > 0 else 0,), dtype="int64", order="F")
        x = np.zeros((config.mdi, config.nm), dtype="float32", order="F")
        xi = np.zeros((0,num_points), dtype="int64", order="F")
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
        agg_iterators = np.zeros((5,config.nmt), dtype="int64", order="F")
        initialize_agg_iterator(config, agg_iterators, sizes_in)
        a_states = np.zeros((config.na, config.ads, config.ans+1), dtype="float32", order="F")
        ay = np.zeros((config.na, config.ado+1), dtype="float32", order="F")
        info = 0
        # Call the routine.
        config.rescale_y = True
        # print()
        # print("config: ", config)
        # print("  ax_in.shape:    ", ax_in.shape)
        # print("  axi_in.shape:   ", axi_in.shape)
        # print("  sizes_in.shape: ", sizes_in.shape)
        # print("  x_in.shape:     ", x_in.shape)
        # print("  xi_in.shape:    ", xi_in.shape)
        # print("  y_in.shape:     ", y_in.shape)
        # print("  yw_in.shape:    ", yw_in.shape)
        # print("  ax.shape:       ", ax.shape)
        # print("  axi.shape:      ", axi.shape)
        # print("  sizes.shape:    ", sizes.shape)
        # print("  x.shape:        ", x.shape)
        # print("  y.shape:        ", y.shape)
        # print("  yw.shape:       ", yw.shape)
        (
            config, model, agg_iterators,
            ax_in, x_in, y_in, yw_in,
            ax, axi, sizes, x, xi, y, yw,
            ax_shift, ax_rescale, axi_shift, axi_rescale, ay_shift,
            x_shift, x_rescale, xi_shift, xi_rescale, y_shift, y_rescale,
            a_emb_vecs, m_emb_vecs,
            a_out_vecs, a_states, ay,
            info
        ) = AXY.normalize_data(
            config, model, agg_iterators,
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
        elif ():
            # TODO: Assert that the mean is near zero and the variance is near one.
            print()
            print("ax_in.shape:\n", ax_in.shape)
            print(" ax_in.mean(axis=1):\n ", ax_in.mean(axis=1))
            print(" ax_in.std(axis=1):\n ", ax_in.std(axis=1))
            print("x_in.shape:\n", x_in.shape)
            print(" x_in.mean(axis=1):\n ", x_in.mean(axis=1))
            print(" x_in.std(axis=1):\n ", x_in.std(axis=1))
            print("y_in.shape:\n", y_in.shape)
            print(" y_in.mean(axis=1): \n ", y_in.mean(axis=1))
            print(" y_in.std(axis=1): \n ", y_in.std(axis=1))
        #  - assert that the AX_IN, X_IN, and Y_IN are radialized
        #  - assert that the AY are radialized (ready for second model)
        #  - assert that the embedded AXI and XI are radialized
        #  - run this routine with inf and NaN inputs
        #  - run this routine with huge amounts of data (make sure memory usage doesn't explode)
        # 

# _test_normalize_data()
# exit()


# 2023-03-13 06:48:18
# 
                #####################################################################################################################################################################################################
                # # TODO: The value test doesn't work with fortran using iterators.                                                                                                                                 #
                # # if ((f_na >= na_in) and (len(sizes_in) == len(sizes))):                                                                                                                                         #
                # #     assert (tuple(ax[:,:na_in].tolist()) == tuple(ax_in[:,:].tolist())), f"AX did not match AX_IN even though there was enough space.\n  python:  {ax_in.tolist()}\n  fortran: {ax.tolist()}\n" #
                #####################################################################################################################################################################################################
