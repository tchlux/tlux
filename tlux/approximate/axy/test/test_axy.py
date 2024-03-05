# TODO:
#  - Identify why (for large categorical aggregate inputs) the initial error would be very large.
#  - Verify the correctness of the gradient when AX and X are both provided.
#  - Create a function for visualizing all of the basis functions in a model.
#  - Make sure the above function works in higher dimension (use PCA?).

try:
    from tlux.approximate.axy.test.test_axy_module import axy as AXY
except ModuleNotFoundError:
    import fmodpy
    # Get the directory for the AXY compiled source code.
    AXY = fmodpy.fimport(
        input_fortran_file = "../axy.f90",
        dependencies = ["pcg32.f90", "axy_profiler.f90", "axy_random.f90", "axy_matrix_operations.f90", "axy_sort_and_select.f90", "axy.f90"],
        name = "test_axy_module",
        blas = True,
        lapack = True,
        omp = True,
        wrap = True,
        # rebuild = True,
        verbose = False,
        f_compiler_args = "-fPIC -shared -O0 -pedantic -fcheck=bounds -ftrapv -ffpe-trap=invalid,overflow,underflow,zero",
    ).axy

# Overwrite the typical "AXY" library with the testing one.
import tlux.approximate.axy.axy
tlux.approximate.axy.axy.axy = AXY

# Overwrite the typical "random" library with the testing one.
from tlux.approximate.axy.test.test_random import random
import tlux.approximate.axy.axy_random
tlux.approximate.axy.axy_random.random = random


# Import codes for testing.
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

import json
import numpy as np
np.set_printoptions(linewidth=1000)


# --------------------------------------------------------------------
#                        SCENARIO_GENERATOR


def _test_scenario_iteration(max_samples=64, show=False):
    print("SCENARIO_GENERATOR")
    import pickle
    # Test scenario generation (for now we only test uniqueness).
    seed = 0
    unique_values = set()
    for i, scenario in enumerate(scenario_generator()):
        # Hash and ensure that we have unique scenarios being generated.
        hash_value = pickle.dumps(scenario)
        if (hash_value in unique_values):
            raise(ValueError(f"A duplicate scenario was generated."))
        unique_values.add(hash_value)
        # Break after generating enough samples.
        if (i >= max_samples): break
        # Skip materializing large data scenarios.
        if (not (scenario["small_data"] or show)): continue
        # Otherwise, attempt to materialize an actual set of data and a model.
        config, details, data, work = gen_config_data(scenario, seed=seed)
        # Print out the scenario if desired.
        if (show):
            print()
            print(i)
            for n in sorted(scenario):
                print(f"  {str(scenario[n]):5s}  {n}")
            print(" data")
            for n in data:
                print(" ", n, data[n].shape if data[n] is not None else data[n])
            print(" temp")
            for n in work:
                print(" ", n, work[n].shape if work[n] is not None else work[n])
    print(" passed")


# --------------------------------------------------------------------
#                           INIT_MODEL

# Test INIT_MODEL
def _test_init_model(show=False):
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
        noe=0, doe=0,  # temporarily disabling output embeddings
    )
    # Initialize the model.
    model = np.zeros(config.total_size, dtype="float32")
    AXY.init_model(config, model, seed=seed,
                   initial_shift_range=initial_shift_range,
                   initial_output_scale=initial_output_scale)
    # Store the model in a format that makes it easy to retrieve vectors.
    from tlux.approximate.axy.summary import AxyModel
    m = AxyModel(config, model)
    # Check that all vectors have a mean of zero, are unique, and have mins and maxes in [-1,1].
    all_values = []
    all_values.extend((m.a_embeddings.T).tolist())
    all_values.extend((m.m_embeddings.T).tolist())
    for i in range(m.a_state_vecs.shape[-1]):
        all_values.extend(m.a_state_vecs[:,:,i].tolist())
    for i in range(m.m_state_vecs.shape[-1]):
        all_values.extend(m.m_state_vecs[:,:,i].tolist())
    all_values = np.asarray(all_values)
    assert (np.all(abs(all_values.mean(axis=0)) < 0.01)), f"Mean values too far from zero."
    assert (np.min(all_values.min(axis=0)) > -1), f"Min values less than -1."
    assert (np.max(all_values.max(axis=0)) < 1), f"Max values greater than 1."
    # Plotting.
    if (show):
        print()
        print(config)
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
    # 
    print(" passed")


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
    #   - partial aggregation
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
        mdo = -1,
        num_threads = 30,
        noe=0, doe=0,  # temporarily disabling output embeddings
    )
    print()
    print(config)
    print()

    # Simple test with nice numbers.
    config.max_batch = 20
    # na = 100
    nm = 100
    # sizes = np.ones(nm, dtype="int32") * (na // nm)
    sizes = np.random.randint(0,10, size=(nm,)).astype("int64")
    na = sum(sizes)
    batcha_starts, batcha_ends, agg_starts, fix_starts, batchm_starts, batchm_ends, info = (
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
    print("fix_starts.shape:    ", fix_starts.shape, fix_starts.tolist())
    print("start diff           ", fix_starts.shape, (fix_starts-agg_starts).tolist())
    print("batchm_starts.shape: ", batchm_starts.shape, batchm_starts.tolist())
    print("batchm_ends.shape:   ", batchm_ends.shape, batchm_ends.tolist())
    print("batchm_starts: ", batchm_starts)
    print("batchm_ends: ", batchm_ends)
    print("batcha_starts: ", batcha_starts)
    print("info: ", info)


# --------------------------------------------------------------------
#                           FETCH_DATA

# Python implementation of an algorithm for clipping the sizes to fit.
#   - start by sorting all sizes,
#   - then add the next largest element
#   - check if all points would fit given that size
#   -- if not, then clip all remaining sets to have the largest fitting size
#   -- if so, subtract from the remaining available space and continue
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
    num_seeds = 5
    na_multitplier = 7
    partial = True
    pairwise = True
    total_seen = 1
    for nm_in in nm_values:
        pairwise = (not pairwise)
        for na_max in range(0, 2*na_range[-1] + 1, na_step):
            partial = (not partial)
            na = na_max * nm_in
            for seed in range(num_seeds):
                # Set a seed.
                np.random.seed(seed)
                # Spawn a new model.
                config, model = spawn_model(adn=1, mdn=1, mdo=1, ade=0)
                config.partial_aggregation = partial
                config.pairwise_aggregation = pairwise
                # Create fake data that matches the shape of the problem.
                sizes_in = np.asarray(np.random.randint(*na_range, size=(nm_in,)), dtype="int64", order="F")
                na_in = sum(sizes_in)
                ax_in = np.asarray(np.random.random(size=(config.adn,na_in)), dtype="float32", order="F")
                axi_in = np.zeros((0,na_in), dtype="int64", order="F")
                x_in = np.asarray(np.random.random(size=(config.mdn,nm_in)), dtype="float32", order="F")
                xi_in = np.zeros((0,nm_in), dtype="int64", order="F")
                y_in = np.asarray(np.random.random(size=(config.mdo,nm_in)), dtype="float32", order="F")
                yi_in = np.zeros((0,nm_in), dtype="int64", order="F")
                yw_in = np.asarray(np.random.random(size=(1,nm_in)), dtype="float32", order="F")
                agg_iterators = np.zeros((6,nm_in), dtype="int64", order="F")
                initialize_agg_iterator(config, agg_iterators, sizes_in, seed=seed)
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
                # Run the size fixing code in python.
                py_sizes, py_na = py_fix_sizes(na, nm, sizes_in[:nm]**(2 if pairwise else 1))
                # Run the size fixing (and data fetching) code in Fortran.
                config = AXY.new_fit_config(nm=nm, nmt=nm_in, na=na, nat=na_in, seed=seed, config=config)
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
                    agg_iterators[5,:] = 0  # iter
                # If we are doing partial aggregation, then make sure that nms is appropriately sized.
                if (config.partial_aggregation):
                    nms = na + nm - 1  # Maxed out when all aggregates belong to one input, the rest have 0.
                else:
                    nms = nm
                # Initialize work space.
                ax = np.zeros((config.adn,na), dtype="float32", order="F")
                axi = np.zeros((0,na), dtype="int64", order="F")
                sizes = np.zeros((nm,), dtype="int64", order="F")
                x = np.zeros((config.mdn,nms), dtype="float32", order="F")
                xi = np.zeros((0,nms), dtype="int64", order="F")
                y = np.zeros((config.mdo,nms), dtype="float32", order="F")
                yi = np.zeros((0,nms), dtype="int64", order="F")
                yw = np.zeros((1,nms), dtype="float32", order="F")
                # Call the Fortran library code.
                total_seen += 1
                (
                    f_config, f_agg_iterators, f_ax, f_axi, f_sizes, f_x, f_xi, f_y, f_yi, f_yw, f_na, f_nm
                ) = AXY.fetch_data(
                    config, agg_iterators, ax_in, ax, axi_in, axi, sizes_in, sizes,
                    x_in, x, xi_in, xi, y_in, y, yi_in, yi, yw_in, yw
                )
                assert (py_na == f_na), f"Number of aggregate points did not match. dict(nm = {nm}, nms = {nms}, seed = {seed})\n  python:  {py_na}\n  fortran: {f_na}\n"
                assert (tuple(sorted(py_sizes.tolist())) == tuple(sorted(f_sizes.tolist()))), f"Sizes did not match. dict(nm = {nm}, nms = {nms}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                assert max(abs(py_sizes-f_sizes)) in {0,1}, f"Sizes did not match. dict(nm = {nm}, nms = {nms}, seed = {seed})\n  python:  {py_sizes}\n  fortran: {f_sizes}\n"
                # Verify that when (nm >= nm_in) and (na >= na_in) that we get ALL inputs!
                if (config.nm >= x_in.shape[1]):
                    # When partial aggregation is on, then repack the data in the expected way.
                    if (config.partial_aggregation):
                        x_in = np.asarray(sum(([x_in[:,i]]*f_sizes[i] for i in range(x_in.shape[1])),[]), dtype=f_x.dtype).T
                        xi_in = np.asarray(sum(([xi_in[:,i]]*f_sizes[i] for i in range(xi_in.shape[1])),[]), dtype=f_xi.dtype).T
                        y_in = np.asarray(sum(([y_in[:,i]]*f_sizes[i] for i in range(y_in.shape[1])),[]), dtype=f_y.dtype).T
                        yw_in = np.asarray(sum(([yw_in[:,i]]*f_sizes[i] for i in range(yw_in.shape[1])),[]), dtype=f_yw.dtype).T
                    assert np.all(x_in == f_x[:,:f_nm]), f"There was enough room for all x, but they were not there (in order).\n\n  sizes {sizes}\n  x:   {x_in}\n  f_x: {f_x}"
                    assert np.all(xi_in == f_xi[:,:f_nm]), f"There was enough room for all xi, but they were not there (in order).\n\n  sizes: {sizes}\n  xi:   {xi_in}\n  f_xi: {f_xi}"
                    assert np.all(y_in == f_y[:,:f_nm]), f"There was enough room for all y, but they were not there (in order).\n\n  sizes {sizes}\n  y:   {y_in}\n  f_y: {f_y}"
                    assert np.all(yw_in == f_yw[:,:f_nm]), f"There was enough room for all yw, but they were not there (in order).\n\n  sizes: {sizes}\n  yw:   {yw_in}\n  f_yw: {f_yw}"
                    if ((not pairwise) and (config.na >= ax_in.shape[1])):
                        assert np.all(sizes_in == f_sizes), f"There was enough room for all sizes, but they were not there (in order).\n\n  sizes:   {sizes_in}\n  f_sizes: {f_sizes}"
                        assert np.all(ax_in == f_ax[:,:ax_in.shape[1]]), f"There was enough room for all ax, but they were not there (in order).\n\n  ax:   {ax_in}\n  f_ax: {f_ax}"
                        assert np.all(axi_in == f_axi[:,:axi_in.shape[1]]), f"There was enough room for all axi, but they were not there (in order).\n\n  axi:   {axi_in}\n  f_axi: {f_axi}"


    print(" passed")




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
    details.agg_iterators_in[0,:] = sizes_in[:]**2 # limit
    details.agg_iterators_in[1,:] = 0 # next
    details.agg_iterators_in[2,:] = 1 # mult
    details.agg_iterators_in[3,:] = 1 # step
    details.agg_iterators_in[4,:] = sizes_in[:]**2 # mod
    details.agg_iterators_in[5,:] = 0 # iter
    print("axi_in: \n", axi_in.T)
    (
        config, agg_iterators, ax, axi, f_sizes, x, xi, y, yw, f_na
    ) = AXY.fetch_data(
        config=config, agg_iterators_in=details.agg_iterators_in, **raw_data, **data,
    )
    print("axi: \n", axi.T)
    # Once we have fetched data that includes pairs, we should verify that they are embedded correctly.
    # 
    # TODO: Verify that the pairs produced by FETCH_DATA match the pairs retrieved by EMBED and EMBEDDING_GRADIENT



# --------------------------------------------------------------------
#                           EVALUATE

from tlux.approximate.axy.axy_py import cast
from tlux.approximate.axy.axy_py import evaluate as py_evaluate

# Fortran evaluation.
#   ax, axi, ay, sizes, x, xi, y, a_states, m_states
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
    assert (data["sizes"].sum() == data["ay"].shape[0]), f"Sizes sum to {data['sizes'].sum()}, a number greater than the length of AY {data['ay'].shape}."
    # Check for a nonzero exit code.
    local_info = 0
    (config, ax, ay, x, y, a_states, m_states, local_info) = AXY.evaluate(
        config, model, info=local_info, **local_data
    )
    check_code(local_info, "AXY.evaluate")
    # Return the dictionary of data values.
    return local_data


def _test_evaluate():
    print("EVALUATE")
    seed = 0
    num_scenarios = 0
    max_scenarios = 100
    for s in scenario_generator(randomized=True, seed=seed):
        # Only consider "small_data" scenarios, others take too long.
        if (not s["small_data"]): continue
        # Increment the number of scenarios seen, break if complete.
        num_scenarios += 1
        if (num_scenarios > max_scenarios):
            break
        scenario_copy = json.loads(json.dumps(s))
        # Materialize the scenario.
        config, details, raw_data, data = gen_config_data(
            seed=seed,
            scenario=s,
        )
        model = details.model
        # Generate a summary in case of failure.
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
        initialize_agg_iterator(config, details.agg_iterators_in, raw_data["sizes_in"], seed=seed)
        # Place a temporary value that is valid into the "SIZES" array.
        if (data["sizes"].size > 0):
            data["sizes"] *= 0
            data["sizes"][0] = config.na
        # Verify the shape of the data.
        info = AXY.check_shape(
            config, model, raw_data["ax_in"], raw_data["axi_in"], raw_data["sizes_in"],
            raw_data["x_in"], raw_data["xi_in"], raw_data["y_in"], raw_data["yi_in"]
        )
        check_code(info, "check_shape")
        # Fetch some raw data into the local placeholders.
        (
            config, details.agg_iterators_in,
            data["ax"], data["axi"], data["sizes"],
            data["x"], data["xi"], data["y"], data["yi"], data["yw"],
            data["na"], data["nm"],
        ) = AXY.fetch_data(
            config=config, agg_iterators_in=details.agg_iterators_in,
            ax_in=raw_data["ax_in"], ax=data["ax"],
            axi_in=raw_data["axi_in"], axi=data["axi"],
            sizes_in=raw_data["sizes_in"], sizes=data["sizes"],
            x_in=raw_data["x_in"], x=data["x"],
            xi_in=raw_data["xi_in"], xi=data["xi"],
            y_in=raw_data["y_in"], y=data["y"],
            yi_in=raw_data["yi_in"], yi=data["yi"],
            yw_in=raw_data["yw_in"], yw=data["yw"],        
        )
        # All necessary keyword arguments for model evaluation.
        na = data["na"]
        nm = data["nm"]
        eval_kwargs = dict(axi=data["axi"][:,:na], ax=data["ax"][:,:na],
                           ay=details.ay[:na,:], sizes=data["sizes"],
                           x=data["x"][:,:nm], xi=data["xi"][:,:nm], y=data["y"][:,:nm],
                           a_states=details.a_states[:na,:,:], m_states=details.m_states[:nm,:,:])
        evaluation = f_evaluate(config, model, **eval_kwargs)
        # Remove the extra matrix for temporary space in the state variables.
        evaluation["a_states"] = evaluation["a_states"][:,:,:-1]
        evaluation["m_states"] = evaluation["m_states"][:,:,:-1]
        # Evaluate with python.
        py = py_evaluate(config, model, **eval_kwargs)
        for key in ("a_states", "ay", "x", "m_states", "y"):
            if (evaluation[key].size == 0): continue
            maxind = np.argmax(np.abs(evaluation[key] - py[key]))
            maxdiff = np.max(np.abs(evaluation[key] - py[key]))
            scenario_string = "{\n  " + ",\n  ".join((f"{repr(k)}: {v}" for (k,v) in scenario_copy.items())) + "\n}"
            assert maxdiff < (2**(-13)), f"ERROR: Failed comparison between Fortran and Python implementations of AXY.evaluate for key '{key}'.\n  Max difference observed was at {maxind}, {maxdiff}.\n  fort:\n{evaluation[key]}\n  python:\n{py[key]}\n\n  Summary of situation:\n\n{summary}\n\nscenario = {scenario_string}"
    # End of for loop.
    print(" passed")



# --------------------------------------------------------------------
#                           MODEL_GRADIENT
def _test_model_gradient():
    print("MODEL_GRADIENT")
    SIMPLIFY_SCENARIO = False
    SHOW_RESULTS = False
    seed = None  # This test is slow, so we do NOT seed it for greater coverage over time.
    num_scenarios = 0
    max_scenarios = 20
    for s in scenario_generator(randomized=True, seed=seed):
        # Only consider "small_data" scenarios, others take too long.
        if (not s["small_data"]): continue
        if (not s["small_model"]): continue
        # TODO: DO not skip categorical outputs, need to update output validation code.
        if (s["output_categorical"]): continue
        # Increment the number of scenarios seen, break if complete.
        num_scenarios += 1
        if (num_scenarios > max_scenarios):
            break
        # # TODO: Each of these scenarios produces a failure because a
        # #       single shift term in the model has a small error
        # #       that is likely due to the nonlinearities near 0.
        # #       Need a way to ignore the one-off isolated errors.
        # elif (num_scenarios in {6,11,14,20}):
        #     continue

        # Simplify the scenario however we want.
        if SIMPLIFY_SCENARIO:
            s.update({
                # 'ade': 2,
                # 'ado': 2,
                'aggregator_only': True,
                'partial_aggregation': True,
                'pairwise_aggregation': False,
                'batch_aggregate_constrained': False,
                'batch_fixed_constrained': False,
                'input_aggregate_categorical': True,
                'input_aggregate_numeric': False,
                'input_fixed_categorical': False,
                'input_fixed_numeric': False,
                'model_aggregate_layered': False,
                'model_fixed_layered': False,
                'output_categorical': False,
                'output_numeric': True,
                'small_data': True,
                'small_model': True,
                'threaded': True,
                'weighted_output': False,
                'weights_dimensioned': False,
                # 'nm_in': 1,
                # 'na_in': 3,
            })
            # Ensure that no more iterations happen after the modified one.
            num_scenarios = max_scenarios
        else:
            # Update the default "na_in" to be a smaller number.
            s["na_in"] = 30

        # Record the scenario for later.
        initial_scenario = s.copy()

        # Create the scenario.
        config, details, raw_data, data = gen_config_data(
            scenario=s,
            seed=seed,
        )
        model = details.model

        # Set things to simple numbers for debugging.
        if SIMPLIFY_SCENARIO:
            details.a_output_vecs[:] = 1.0
            details.m_output_vecs[:] = 1.0
            raw_data["ax_in"][:] = 1.0
            raw_data["y_in"][:] = 2.0
            details.ay_shift *= 0.0
            # Print out the shapes of raw data.
            print("Raw data")
            for (k,v) in sorted(raw_data.items()):
                print("", k, v.shape if hasattr(v, "shape") else v)
            print()
            print("destination")
            for (k,v) in sorted(data.items()):
                print("", k, v.shape if hasattr(v, "shape") else v)
            print()


        # Fetch and embed data (for model evaluation).
        initialize_agg_iterator(config, details.agg_iterators_in, raw_data["sizes_in"], seed=seed)
        (
            config, details.agg_iterators_in,
            data["ax"], data["axi"], data["sizes"],
            data["x"], data["xi"], data["y"], data["yi"], data["yw"],
            data["na"], data["nm"],
        ) = AXY.fetch_data(
            config=config, agg_iterators_in=details.agg_iterators_in,
            ax_in=raw_data["ax_in"], ax=data["ax"],
            axi_in=raw_data["axi_in"], axi=data["axi"],
            sizes_in=raw_data["sizes_in"], sizes=data["sizes"],
            x_in=raw_data["x_in"], x=data["x"],
            xi_in=raw_data["xi_in"], xi=data["xi"],
            y_in=raw_data["y_in"], y=data["y"],
            yi_in=raw_data["yi_in"], yi=data["yi"],
            yw_in=raw_data["yw_in"], yw=data["yw"],        
        )

        if SIMPLIFY_SCENARIO:
            print("Data:")
            for (k,v) in data.items():
                print("", f"{k:>5s}", v.shape if hasattr(v, "shape") else v, end="")
                if (k == "sizes"):
                    print(f" {v.tolist()}")
                else:
                    print()
            print()

        # Shrink the "x" and "y" values according to the size of the fetched data.
        na = data["na"]
        nm = data["nm"]
        eval_kwargs = dict(
            axi=data["axi"][:,:na], ax=data["ax"][:,:na],
            ay=details.ay[:na,:], sizes=data["sizes"],
            x=data["x"][:,:nm], xi=data["xi"][:,:nm],
            y=data["y"][:,:nm], yi=data["yi"][:,:nm], yw=data["yw"][:,:nm],
            a_states=details.a_states[:na,:,:], m_states=details.m_states[:nm,:,:],
            na=na, nm=nm,
        )

        # Approximate the gradient with a finite difference.
        from tlux.math.fraction import Fraction
        dtype = Fraction
        # dtype = "float64"
        offset = Fraction(1, 2**52)
        def finite_difference_gradient(model, data, config=config, offset=offset, dtype=dtype):
            na, nm = data["na"], data["nm"]
            ax, axi = data["ax"].copy(order="F"), data["axi"].copy(order="F")
            sizes = data["sizes"].copy(order="F")
            x, xi = data["x"].copy(order="F"), data["xi"].copy(order="F")
            y, yw = cast(data["y"], dtype), cast(data["yw"], dtype)
            model = cast(model, dtype)
            # Translate the "yw" values the same as in the AXY code.
            yw = np.where(yw < 0, 1 / (1 + abs(yw)), yw)
            n = y.shape[1]
            # Evaluate the model and compute the current error.
            f = py_evaluate(config, model, dtype=dtype, ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)
            fy = f["y"]
            squared_error = (fy - y)**2 / 2
            # Initialize the gradient to zero.
            gradient = 0 * model.copy()[:config.num_vars]
            # For each component of the model, modify the model, evaluate, observe change in error.
            for i in range(config.num_vars):
                local_model = model.copy()
                local_model[i] += offset
                local_fy = py_evaluate(config, local_model, dtype=dtype, **data)["y"]
                local_squared_error = (local_fy - y)**2 / 2
                # Compute the error.
                error_delta = (local_squared_error - squared_error) / n
                # Handle weighted outputs.
                if (yw.shape[0] == 1):
                    error_delta = (error_delta.T * yw.T).T
                elif (yw.shape[0] == y.shape[0]):
                    error_delta = error_delta * yw
                # Store the gradient for this model variable.
                gradient[i] = error_delta.sum() / offset
            # Return the gradient (in place of the model variables).
            return np.concatenate((gradient, model[config.num_vars:]))

        # Compute the gradient with the library function.
        def axy_gradient(model, data, config=config, details=details):
            # Define the copy of the data that will be used for evaluating the model.
            info = 0
            # The "na" key should contain the actual number of aggregate values (after fetch).
            na, nm = data["na"], data["nm"]
            ax, axi = data["ax"].copy(order="F"), data["axi"].copy(order="F")
            sizes = data["sizes"].copy(order="F")
            x, xi = data["x"].copy(order="F"), data["xi"].copy(order="F")
            y, yi, yw = data["y"].copy(order="F"), data["yi"].copy(order="F"), data["yw"].copy(order="F")
            # Extract data from "details" (those need to be size-adjusted).
            ay = details.ay[:na,:].copy(order="F")
            a_states = details.a_grads[:na,:,:].copy(order="F")
            m_states = details.m_grads[:nm,:,:].copy(order="F")
            # Evaluate the model.
            evaluation = f_evaluate(config, model, ax=ax, axi=axi, ay=ay, sizes=sizes, x=x, xi=xi, y=y,
                                    a_states=a_states, m_states=m_states)
            y_gradient = evaluation["y"]
            m_grads = evaluation["m_states"]
            x = evaluation["x"]
            ay_gradient = evaluation["ay"]
            a_grads = evaluation["a_states"]
            ax = evaluation["ax"]
            check_code(info, "AXY.evaluate")
            model_grad = 0 * details.model_grad
            a_emb_temp = 0 * details.a_emb_temp
            m_emb_temp = 0 * details.m_emb_temp
            o_emb_temp = 0 * details.o_emb_temp
            emb_outs = details.emb_outs
            emb_grads = details.emb_grads
            (
                config,
                ax,
                x_gradient,
                sum_squared_gradient,
                model_grad,
                info,
                ay_gradient,
                y_gradient,
                a_grads,
                m_grads,
                a_emb_temp,
                m_emb_temp,
                o_emb_temp,
                emb_outs,
                emb_grads,
            ) = AXY.model_gradient(
                config, model,
                ax=ax, axi=axi, sizes=sizes,
                x=x, xi=xi,
                y=y, yi=yi, yw=yw,
                sum_squared_gradient=0.0,
                model_grad=model_grad,
                info=info,
                ay_gradient=ay_gradient,
                y_gradient=y_gradient,
                a_grads=a_grads,
                m_grads=m_grads,
                a_emb_temp=a_emb_temp,
                m_emb_temp=m_emb_temp,
                o_emb_temp=o_emb_temp,
                emb_outs=emb_outs,
                emb_grads=emb_grads,
            )
            check_code(info, "AXY.model_gradient")
            # TODO: Update the gradient checks here to calculate the AY error term as well.
            if (config.asov <= config.aeov):
                # Overwrite the gradient for the AY error term to be zero.
                model_grad[config.asov-1:config.aeov,:].reshape(config.adso, config.ado+1, -1, order="F")[:,-1,:] = 0.0
            # div = min(model_grad.shape[1], y.shape[1])
            return np.concatenate((model_grad.sum(axis=1), model[config.num_vars:]))

        gradient = finite_difference_gradient(model, eval_kwargs)
        model_gradient = axy_gradient(model, eval_kwargs)
        error = (model_gradient - gradient)
        ratio = np.asarray([1 + guess if (val == 0) else guess / val
                            for (guess,val) in zip(model_gradient, gradient)])
        # max_error = abs(error).max()
        # max_ratio_error = abs(ratio-1).max()
        # failed_test = (max_error > 0.0001) and (max_ratio_error > 0.01)
        max_error = abs(error).mean()
        max_ratio_error = abs(ratio-1).mean()
        failed_test = (max_error > 0.0003) and (max_ratio_error > 0.01)

        if (failed_test or SHOW_RESULTS):
            print()
            scenario_string = "{\n  " + ",\n  ".join((f"{repr(k)}: {v}" for (k,v) in initial_scenario.items())) + "\n}"
            print("scenario =", scenario_string)
            print()
            print("AxyModel(config, model): ")
            # print(AxyModel(config, model, show_vecs=True, show_times=False))
            print("error: ", error[np.argsort(-abs(error))[:5]].astype("float32"))
            print("ratio: ", ratio[np.argsort(abs(ratio-1))[:5]].astype("float32"))
            print("max_error:       ", float(max_error))
            print("max_ratio_error: ", float(max_ratio_error))
            print()
            # Show the configuration and the data.
            print()
            print("-"*100)
            print("config: ", config)
            print()
            print("Data:")
            for (k,v) in data.items():
                print("", type(k).__name__, k, v.shape if hasattr(v,"shape") else "")
            print("-"*100)
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
            scenario_string = "{\n  " + ",\n  ".join((f"{repr(k)}: {v}" for (k,v) in initial_scenario.items())) + "\n}"
            assert (not failed_test), f"Either the maximum error ({float(max_error)}) was too high or the ratio between the exact gradient and computed gradient ({float(max_ratio_error)}) was too far from 1.\n\nscenario = {scenario_string}"

    # Only arrives here if no tests fail.
    print(" passed")



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
    p.add("axi before", *axi.T, marker_size=2)
    data.pop("yi_in", None)

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
    p.add("axi after", *axi.T, marker_size=2)

    p.show()



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
def _test_normalize_data(dimension=64, show=False):
    print("NORMALIZE_DATA")
    # Generate data to test with.
    num_trials = 5
    size_range = (0, 1000)
    np.random.seed(0)
    num_points = 500
    nm = 250
    na = 300
    should_plot = (dimension <= 3) and (na + nm < 100000) and (show)
    if (should_plot):
        from tlux.plot import Plot
        p = Plot()
    for i in range(num_trials):
        # Generate random data (that is skewed and placed off center).
        x, x_shift, x_scale, x_rotation = random_data(num_points, dimension)
        if (should_plot):
            p.add("x "+str(i+1), *x.T, marker_size=3)
        y, y_shift, y_scale, y_rotation = random_data(num_points, dimension)
        if (should_plot):
            p.add("y "+str(i+1), *y.T, marker_size=3)
        # Generate random AX data.
        sizes = np.asarray(np.random.randint(*size_range, size=(num_points,)), dtype="int64", order="F")
        num_aggregate = sizes.sum()
        ax, ax_shift, ax_scale, ax_rotation = random_data(num_aggregate, dimension)
        if (should_plot):
            p.add("ax "+str(i+1), *ax.T, marker_size=3)
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
        yi_in = np.zeros((0,num_points), dtype="int64", order="F")
        yw_in = np.zeros((0,num_points), dtype="float32", order="F")
        # Set up the "data holder".
        ax = np.zeros((config.adn,config.na), dtype="float32", order="F")
        axi = np.zeros((0,config.na), dtype="int64", order="F")
        sizes = np.zeros((config.nm if config.na > 0 else 0,), dtype="int64", order="F")
        x = np.zeros((config.mdi, config.nm), dtype="float32", order="F")
        xi = np.zeros((0,num_points), dtype="int64", order="F")
        y = np.zeros((config.do, config.nm), dtype="float32", order="F")
        yi = np.zeros((0,num_points), dtype="int64", order="F")
        yw = np.zeros((0,num_points), dtype="float32", order="F")
        # Set up the "transformations".
        ax_shift = np.zeros((config.adn,), dtype="float32", order="F")
        ax_rescale = np.zeros((config.adn,config.adn), dtype="float32", order="F")
        axi_shift = np.zeros((config.ade,), dtype="float32", order="F")
        axi_rescale = np.zeros((config.ade,config.ade), dtype="float32", order="F")
        ay_shift = np.zeros((config.ado,), dtype="float32", order="F")
        ay_scale = np.zeros((config.ado,), dtype="float32", order="F")
        x_shift = np.zeros((config.mdn,), dtype="float32", order="F")
        x_rescale = np.zeros((config.mdn,config.mdn), dtype="float32", order="F")
        xi_shift = np.zeros((config.mde,), dtype="float32", order="F")
        xi_rescale = np.zeros((config.mde,config.mde), dtype="float32", order="F")
        y_shift = np.zeros((config.do,), dtype="float32", order="F")
        y_rescale = np.zeros((config.do,config.do), dtype="float32", order="F")
        yi_shift = np.zeros((config.doe,), dtype="float32", order="F")
        yi_rescale = np.zeros((config.doe,config.doe), dtype="float32", order="F")
        # Set up the remaining data matrices.
        a_emb_vecs = model[config.asev-1:config.aeev].reshape((config.ane, config.ade)).T
        m_emb_vecs = model[config.msev-1:config.meev].reshape((config.mne, config.mde)).T
        o_emb_vecs = model[config.osev-1:config.oeev].reshape((config.doe, config.doe)).T
        a_out_vecs = model[config.asov-1:config.aeov].reshape((config.ado+1, config.adso)).T
        agg_iterators = np.zeros((6,config.nmt), dtype="int64", order="F")
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
            ax, axi, sizes, x, xi, y, yi, yw,
            ax_shift, ax_rescale, axi_shift, axi_rescale, ay_shift, ay_scale,
            x_shift, x_rescale, xi_shift, xi_rescale,
            y_shift, y_rescale, yi_shift, yi_rescale,
            a_emb_vecs, m_emb_vecs, o_emb_vecs,
            a_out_vecs, a_states, ay,
            info
        ) = AXY.normalize_data(
            config, model, agg_iterators,
            ax_in, axi_in, sizes_in, x_in, xi_in, y_in, yi_in, yw_in,
            ax, axi, sizes, x, xi, y, yi, yw,
            ax_shift, ax_rescale, axi_shift, axi_rescale, ay_shift, ay_scale,
            x_shift, x_rescale, xi_shift, xi_rescale,
            y_shift, y_rescale, yi_shift, yi_rescale,
            a_emb_vecs, m_emb_vecs, o_emb_vecs,
            a_out_vecs, a_states, ay, info
        )
        if (should_plot):
            p.add("-> x "+str(i+1), *x_in, marker_size=3)
            p.add("-> y "+str(i+1), *y_in, marker_size=3)
            p.add("-> ax "+str(i+1), *ax_in, marker_size=3)
            p.show()
        elif (True):
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
            exit()
        #  - assert that the AX_IN, X_IN, and Y_IN are radialized
        #  - assert that the AY are radialized (ready for second model)
        #  - assert that the embedded AXI and XI are radialized
        #  - run this routine with inf and NaN inputs
        #  - run this routine with huge amounts of data (make sure memory usage doesn't explode)
        # 
    print(" passed")


def _test_condition_model():
    print("CONDITION_MODEL")
    seed = 0
    num_scenarios = 0
    max_scenarios = 1
    for s in scenario_generator(randomized=True, seed=seed):
        # Only consider "small_data" scenarios, others take too long.
        if (not s["small_data"]): continue
        # Increment the number of scenarios seen, break if complete.
        num_scenarios += 1
        if (num_scenarios > max_scenarios):
            break
        scenario_copy = json.loads(json.dumps(s))
        # Materialize the scenario.
        config, details, raw_data, data = gen_config_data(
            seed=seed,
            scenario=s,
            nm=100,
        )
        # Extract all the arguments and call the routine.
        model = details.model
        model_grad = details.model_grad
        model_grad_mean = details.model_grad_mean
        model_grad_curv = details.model_grad_curv
        ax = details.ax
        axi = details.axi
        ay = details.ay
        ay_gradient = details.ay_gradient
        sizes = details.sizes
        x_gradient = details.x_gradient
        x = details.x
        xi = details.xi
        y = details.y
        yi = details.yi
        yw = details.yw
        y_gradient = details.y_gradient
        num_threads = details.config.num_threads
        fit_step = details.config.steps_taken
        a_states = details.a_states
        m_states = details.m_states
        a_grads = details.a_grads
        m_grads = details.m_grads
        a_lengths = details.a_lengths
        m_lengths = details.m_lengths
        a_emb_temp = details.a_emb_temp
        m_emb_temp = details.m_emb_temp
        o_emb_temp = details.o_emb_temp
        emb_outs = details.emb_outs
        emb_grads = details.emb_grads
        a_state_temp = details.a_state_temp
        m_state_temp = details.m_state_temp
        a_order = details.a_order
        m_order = details.m_order
        update_indices = details.update_indices
        total_eval_rank = 0
        total_grad_rank = 0
        sum_squared_error = 0.0
        info = 0
        step_mean_remain = 1.0 - config.step_mean_change
        step_curv_remain = 1.0 - config.step_curv_change
        # Make the data normalized, zero mean, unit variance, and varied embeddings.
        ax[:,:] = np.random.normal(0.0, 1.0, size=ax.shape)
        axi[:,:] = np.random.randint(1, config.ane, size=axi.shape)
        x[:,:] = np.random.normal(0.0, 1.0, size=x.shape)
        xi[:,:] = np.random.randint(1, config.mne, size=xi.shape)
        y[:,:] = np.random.normal(0.0, 1.0, size=y.shape)
        # Make sure that all the destinations for the embeddings are zeroed out.
        ax[:,config.adn:] = 0.0
        x[:,config.mdn:] = 0.0
        # Embed data and evaluate the model.
        AXY.embed(config, model, axi, xi, ax, x)
        AXY.evaluate(config, model, ax, ay_gradient, sizes, x_gradient, y, a_grads, m_grads, info)
        a_states[:,:,:] = a_grads[:,:,:]
        m_states[:,:,:] = m_grads[:,:,:]        
        ay[:,:] = ay_gradient[:,:]
        x[:,:] = x_gradient[:,:]
        print("config: ", config, flush=True)
        print()
        print("Values before gradient calculation:")
        print("  ax.mean(axis=1):", ax.mean(axis=1)[config.adn:], flush=True)
        print("  ax.std(axis=1): ", ax.std(axis=1)[config.adn:], flush=True)
        print("  x.mean(axis=1): ", x.mean(axis=1)[config.mdn:], flush=True)
        print("  x.std(axis=1):  ", x.std(axis=1)[config.mdn:], flush=True)
        print()
        # Compute gradient, adjust step sizes, take model step.
        AXY.model_gradient(config, model, ax, axi, sizes, x_gradient, xi, y, yi, yw,
                           sum_squared_error, model_grad, info,
                           ay_gradient, y_gradient,
                           a_grads, m_grads, a_emb_temp, m_emb_temp, o_emb_temp,
                           emb_outs, emb_grads)
        # ---------------------------------------------------------------------------------
        # AXY.adjust_rates(model, model_grad_mean, model_grad_curv)
        # AXY.step_variables(model_grad, model_grad_mean, model_grad_curv, update_indices, num_threads)
        # 
        # Aggregate over computed batches and compute average gradient.
        model_grad[:, 0] = np.mean(model_grad[:, :], axis=1)
        # Mean.
        model_grad_mean[:] = step_mean_remain * model_grad_mean[:] + config.step_mean_change * model_grad[:, 0]
        # Clip the mean to be small enough to be numerically stable.
        model_grad_mean[:] = np.where(np.abs(model_grad_mean[:]) > config.max_step_component,
                                      np.sign(config.max_step_component, model_grad_mean[:]),
                                      model_grad_mean[:])
        # Curvature.
        model_grad_curv[:] = step_curv_remain * model_grad_curv[:] + \
                             config.step_curv_change * (model_grad_mean[:] - model_grad[:, 0])**2
        # Clip the curvature to be large enough to be numerically stable.
        model_grad_curv[:] = np.where(model_grad_curv[:] < config.min_curv_component,
                                      config.min_curv_component,
                                      model_grad_curv[:])
        # Clip the curvature to be small enough to be numerically stable.
        model_grad_curv[:] = np.where(np.abs(model_grad_curv[:]) > config.max_curv_component,
                                      np.sign(config.max_curv_component, model_grad_curv[:]),
                                      model_grad_curv[:])
        # Set the step as the mean direction (over the past few steps).
        model_grad[:, 0] = model_grad_mean[:]
        # Start scaling by step magnitude by curvature once enough data is collected.
        model_grad[:, 0] /= np.sqrt(model_grad_curv[:])
        # Take the gradient steps (based on the computed "step" above).
        model[:config.num_vars] -= model_grad[:, 0] * config.step_factor
        # ---------------------------------------------------------------------------------
        print("Values after gradient computation, before conditioning:")
        print("  ax.mean(axis=1):", ax.mean(axis=1)[config.adn:], flush=True)
        print("  ax.std(axis=1): ", ax.std(axis=1)[config.adn:], flush=True)
        print("  x.mean(axis=1): ", x.mean(axis=1)[config.mdn:], flush=True)
        print("  x.std(axis=1):  ", x.std(axis=1)[config.mdn:], flush=True)
        print()
        # Now trigger the conditioning operation.
        result = AXY.condition_model(
            config, model, model_grad_mean, model_grad_curv,
            ax, axi, ay, ay_gradient, sizes, x, x_gradient, xi, y,
            y_gradient, num_threads, fit_step,
            a_states, m_states, a_grads, m_grads,
            a_lengths, m_lengths, a_state_temp, m_state_temp,
            a_order, m_order, total_eval_rank, total_grad_rank
        )
        print("result[-1]: ", result[-1], flush=True)        
        print("result[-2]: ", result[-2], flush=True)
        print()
        # ---------------------------------------
        # Embed data and evaluate the model.
        AXY.embed(config, model, axi, xi, ax, x)
        AXY.evaluate(config, model, ax, ay, sizes, x, y, a_grads, m_grads, info)
        a_states[:,:,:] = a_grads[:,:,:]
        m_states[:,:,:] = m_grads[:,:,:]        
        ay[:,:] = ay_gradient[:,:]
        print("Values after conditioning and re-evaluation:")
        print("  ax.mean(axis=1):", ax.mean(axis=1)[config.adn:], flush=True)
        print("  ax.std(axis=1): ", ax.std(axis=1)[config.adn:], flush=True)
        print("  x.mean(axis=1): ", x.mean(axis=1)[config.mdn:], flush=True)
        print("  x.std(axis=1):  ", x.std(axis=1)[config.mdn:], flush=True)
        print()
    print(" passed")


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
    print(" data")
    for n in data:
        print(" ", n, data[n].shape if data[n] is not None else data[n])
    print(" temp")
    for n in work:
        print(" ", n, work[n].shape if work[n] is not None else work[n])
    print(" config")
    print(" ", config)
    (
        config, model, rwork, iwork, lwork, ax_in, x_in, y_in, yw_in,
        record, sum_squared_error, info
    ) = AXY.fit_model(
        config, model, rwork, iwork, lwork,
        data["ax_in"], data["axi_in"], data["sizes_in"],
        data["x_in"], data["xi_in"], data["y_in"], data["yw_in"],
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


def _test_normalize_step():
    print("NORMALIZE_STEP")
    # 
    from tlux.plot import Plot
    p = Plot()
    # Generate data to test with.
    np.random.seed(0)
    size_range = (0, 10)
    num_points = 100
    dimension = 3
    nm = num_points
    i = 0
    # Generate random data (that is skewed and placed off center).
    x, x_shift, x_scale, x_rotation = random_data(num_points, dimension)
    p.add("x "+str(i+1), *x.T, marker_size=3)
    y, y_shift, y_scale, y_rotation = random_data(num_points, dimension)
    p.add("y "+str(i+1), *y.T, marker_size=3)
    # Generate random AX data.
    sizes = np.asarray(np.random.randint(*size_range, size=(num_points,)), dtype="int64", order="F")
    num_aggregate = sizes.sum()
    na = num_aggregate
    ax, ax_shift, ax_scale, ax_rotation = random_data(num_aggregate, dimension)
    p.add("ax "+str(i+1), *ax.T, marker_size=3)
    # Set up all of the inputs to the NORMALIZE_DATA routine.
    config, model = spawn_model(adn=dimension, mdn=dimension, mdo=dimension, num_threads=1)
    config.pairwise_aggregation = False
    config = AXY.new_fit_config(nm=nm, nmt=num_points, na=na, nat=num_aggregate, config=config)
    rwork = np.ones(config.rwork_size, dtype="float32", order="F")
    # Set YW"
    yw = np.zeros((0,num_points), dtype="float32", order="F")
    # Call the step normalization function.
    AXY.normalize_step(config, model, rwork, ax.T, sizes, x.T, y.T, yw)
    p.show(file_name="/tmp/test_normalize_step.html")
    print(" passed")


if __name__ == "__main__":
    # _test_normalize_step()
    # exit()
    _test_scenario_iteration()
    _test_init_model()
    # _test_compute_batches() # TODO: Design this test more carefully.
    _test_fetch_data()
    # _test_embed() # TODO: Design this test.
    # _test_normalize_data() # TODO: Test with assertions above 3D.
    # _test_condition_model() # TODO: Formalize this test to have assertions.
    _test_evaluate()
    _test_model_gradient()
    # 
    # _test_large_data_fit()
    # _test_axi()
