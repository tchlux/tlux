# This file defines possible model scenarios that need to be considered
#  when testing. It provides functions for generating unique scenarios
#  with model, data, and work allocations, as well as an iterator that
#  exhaustively iterates over all possible data and model scenarios.

import fmodpy

# Most combinations of these define the possible uses of the model.
SCENARIO = dict(
    steps = 50,
    ade = None,
    ado = None,
    mde = None,
    aggregator_only = False,
    aggregate_pairwise = False,
    batch_aggregate_constrained = False,
    batch_fixed_constrained = True,
    input_aggregate_categorical = True,
    input_aggregate_numeric = False,
    input_fixed_categorical = False,
    input_fixed_numeric = False,
    model_aggregate_layered = False,
    model_fixed_layered = False,
    output_categorical = False,
    output_numeric = True,
    small_data = False,
    small_model = True,
    threaded = True,
    weighted_output = False,
    weights_dimensioned = False,
)


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


# Holder for the model.
class Details(dict):
    def __init__(self, config, steps, ydi=0, ywd=0):
        import numpy as np
        self.config = config
        self.steps = steps
        # Modify nm and na to be the smallest needed values.
        # Generic allocations and objects.
        ftype = dict(order="F", dtype="float32")
        itype = dict(order="F", dtype="int32")
        ltype = dict(order="F", dtype="int64")
        model = np.ones(config.total_size, **ftype)
        rwork = np.ones(config.rwork_size, **ftype)  # beware of allocation, heap vs stack
        iwork = np.ones(config.iwork_size, **itype)
        agg_iterators = np.ones((5, config.nmt), **ltype)
        record = np.zeros((6,steps), **ftype)
        yi = np.zeros((ydi, config.nm), **itype)
        yw = np.zeros((ywd, config.nm), **ftype)
        # Store source memory allocations internally.
        self.model = model
        self.rwork = rwork
        self.iwork = iwork
        self.record = record
        self.agg_iterators = agg_iterators
        # Declare all the special attributes.
        self.update(dict(
            # Model.
            a_embeddings  = model[config.asev-1:config.aeev].reshape(config.ade, config.ane, order="F"),
            a_input_vecs  = model[config.asiv-1:config.aeiv].reshape(config.adi, config.ads, order="F"),
            a_input_shift = model[config.asis-1:config.aeis].reshape(config.ads, order="F"),
            a_state_vecs  = model[config.assv-1:config.aesv].reshape(config.ads, config.ads, max(0,config.ans-1), order="F"),
            a_state_shift = model[config.asss-1:config.aess].reshape(config.ads, max(0,config.ans-1), order="F"),
            a_output_vecs = model[config.asov-1:config.aeov].reshape(config.adso, config.ado+1, order="F"),
            m_embeddings  = model[config.msev-1:config.meev].reshape(config.mde, config.mne, order="F"),
            m_input_vecs  = model[config.msiv-1:config.meiv].reshape(config.mdi, config.mds, order="F"),
            m_input_shift = model[config.msis-1:config.meis].reshape(config.mds, order="F"),
            m_state_vecs  = model[config.mssv-1:config.mesv].reshape(config.mds, config.mds, max(0,config.mns-1), order="F"),
            m_state_shift = model[config.msss-1:config.mess].reshape(config.mds, max(0,config.mns-1), order="F"),
            m_output_vecs = model[config.msov-1:config.meov].reshape(config.mdso, config.mdo, order="F"),
            ax_shift = model[config.aiss-1:config.aise].reshape(config.adn, order="F"),
            ax_rescale = model[config.aims-1:config.aime].reshape(config.adn, config.adn, order="F"),
            ay_shift = model[config.aoss-1:config.aose].reshape(config.ado, order="F"),
            x_shift = model[config.miss-1:config.mise].reshape(config.mdn, order="F"),
            x_rescale = model[config.mims-1:config.mime].reshape(config.mdn, config.mdn, order="F"),
            y_shift = model[config.moss-1:config.mose].reshape(config.do, order="F"),
            y_rescale = model[config.moms-1:config.mome].reshape(config.do, config.do, order="F"),
            # Real work space.
            model_grad = rwork[config.smg-1:config.emg].reshape(config.num_vars, config.num_threads, order="F"),
            model_grad_mean = rwork[config.smgm-1:config.emgm].reshape(config.num_vars, order="F"),
            model_grad_curv = rwork[config.smgc-1:config.emgc].reshape(config.num_vars, order="F"),
            best_model = rwork[config.sbm-1:config.ebm].reshape(config.num_vars, order="F"),
            ax = rwork[config.saxb-1:config.eaxb].reshape(config.adi, config.na, order="F"),
            a_emb_temp = rwork[config.saet-1:config.eaet].reshape(config.ade, config.ane, config.num_threads, order="F"),
            a_emb_counts = rwork[config.saec-1:config.eaec].reshape(config.ane, config.num_threads, order="F"),
            a_states = rwork[config.saxs-1:config.eaxs].reshape(config.na, config.ads, config.ans+1, order="F"),
            a_grads = rwork[config.saxg-1:config.eaxg].reshape(config.na, config.ads, config.ans+1, order="F"),
            ay = rwork[config.say-1:config.eay].reshape(config.na, config.ado+1, order="F"),
            ay_gradient = rwork[config.sayg-1:config.eayg].reshape(config.na, config.ado+1, order="F"),
            x = rwork[config.smxb-1:config.emxb].reshape(config.mdi, config.nm, order="F"),
            m_emb_temp = rwork[config.smet-1:config.emet].reshape(config.mde, config.mne, config.num_threads, order="F"),
            m_emb_counts = rwork[config.smec-1:config.emec].reshape(config.mne, config.num_threads, order="F"),
            m_states = rwork[config.smxs-1:config.emxs].reshape(config.nm, config.mds, config.mns+1, order="F"),
            m_grads = rwork[config.smxg-1:config.emxg].reshape(config.nm, config.mds, config.mns+1, order="F"),
            y = rwork[config.smyb-1:config.emyb].reshape(config.do, config.nm, order="F"),
            y_gradient = rwork[config.syg-1:config.eyg].reshape(config.do, config.nm, order="F"),
            axi_shift = rwork[config.saxis-1:config.eaxis].reshape(config.ade, order="F"),
            axi_rescale = rwork[config.saxir-1:config.eaxir].reshape(config.ade, config.ade, order="F"),
            xi_shift = rwork[config.smxis-1:config.emxis].reshape(config.mde, order="F"),
            xi_rescale = rwork[config.smxir-1:config.emxir].reshape(config.mde, config.mde, order="F"),
            a_lengths = rwork[config.sal-1:config.eal].reshape(config.ads, config.num_threads, order="F"),
            m_lengths = rwork[config.sml-1:config.eml].reshape(config.mds, config.num_threads, order="F"),
            a_state_temp = rwork[config.sast-1:config.east].reshape(config.na, config.ads, order="F"),
            m_state_temp = rwork[config.smst-1:config.emst].reshape(config.nm, config.mds, order="F"),
            # Integer work space.
            axi = iwork[config.saxi-1:config.eaxi].reshape(-1, config.na, order="F"),
            xi = iwork[config.smxi-1:config.emxi].reshape(-1, config.nm, order="F"),
            sizes = iwork[config.ssb-1:config.esb].reshape(config.nm, order="F"),
            a_order = iwork[config.sao-1:config.eao].reshape(config.ads, config.num_threads, order="F"),
            m_order = iwork[config.smo-1:config.emo].reshape(config.mds, config.num_threads, order="F"),
            # External space.
            yi = yi,
            yw = yw,
        ))

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)
    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)


# Check the exit code values.
def check_code(exit_code, method):
    import os, re
    if (exit_code != 0):
        # Read the reason from the Fortran source file.
        source = ""
        source_file = "../axy.f90"
        if (os.path.exists(source_file)):
            with open(source_file, "r") as f:
                source = f.read()
        reason = re.search(r"INFO\s*=\s*" + str(exit_code) + r"\s*[!][^\n]*\n", source)
        if (reason is not None):
            reason = reason.group().strip()
            reason = reason[reason.index("!")+1:].strip()
        else:
            reason = ""
        # Raise an assertion error.
        assert (exit_code == 0), f"AXY.{method} returned nonzero exit code {exit_code}. {reason}"


# Spawn a new model that is ready to be evaluated.
def spawn_model(adn, mdn, mdo, ade=0, ane=0, ads=3, ans=2, ado=None, mde=0, mne=0, mds=3, mns=2,
                seed=0, num_threads=30, initial_shift_range=1.0, initial_output_scale=0.1):
    import numpy as np
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


# Given a scenario, generate a model and data to match the scenario.
def gen_config_data(scenario=None, seed=0, default=SCENARIO, **scenario_kwargs):
    # Update the default scenario with the provided one.
    settings = default.copy()
    if (scenario is not None):
        settings.update(scenario)
    scenario = settings
    scenario.update(scenario_kwargs)
    # Problem size.
    if scenario["small_data"]:
        nm_in = scenario.get('nm_in', 10)
        na_in = scenario.get('na_in', 100)
        na_range = scenario.get('na_range', (1, 5))
        nm_range = scenario.get('nm_range', (1, 5))
    else:
        nm_in = scenario.get('nm_in', 10000)
        na_in = scenario.get('na_in', 1000000)
        na_range = scenario.get('na_range', (1, 5000))
        nm_range = scenario.get('nm_range', (1, 5000))
    # Fixed batch constrained (cannot fit all fixed data in at once).
    if scenario["batch_fixed_constrained"]:
        nm = scenario.get('nm', nm_in // 5)
    else:
        nm = scenario.get('nm', nm_in)
    # Aggregate batch constrained (cannot fit all aggregate data in at once).
    if scenario["batch_aggregate_constrained"]:
        na = scenario.get('na', na_in // 5)
    else:
        na = scenario.get('na', na_in)
    # Aggregate numeric input.
    if scenario["input_aggregate_numeric"]:
        if scenario["small_data"]:
            adn = scenario.get('adn', 2)
        else:
            adn = scenario.get('adn', 20)
    else:
        adn = 0 # scenario.get('adn', 0)
    # Aggregate categorical input.
    if scenario["input_aggregate_categorical"]:
        if scenario["small_data"]:
            ane = scenario.get('ane', 5)
            adi = scenario.get('adi', 1)
        else:
            ane = scenario.get('ane', 5000)
            adi = scenario.get('adi', 5)
    else:
        ane = 0 # scenario.get('ane', 0)
        adi = 0 # scenario.get('adi', 0)
    # Aggregate model layered.
    if scenario["model_aggregate_layered"]:
        if scenario["small_model"]:
            ans = scenario.get('ans', 2)
            ads = scenario.get('ads', 3)
        else:
            ans = scenario.get('ans', 20)
            ads = scenario.get('ads', 64)
    else:
        ans = scenario.get('ans', 0)
        ads = scenario.get('ads', 0)
    # Fixed numeric input.
    if scenario["input_fixed_numeric"]:
        if scenario["small_data"]:
            mdn = scenario.get('mdn', 2)
        else:
            mdn = scenario.get('mdn', 20)
    else:
        mdn = scenario.get('mdn', 0)
    # Fixed categorical input.
    if scenario["input_fixed_categorical"]:
        if scenario["small_data"]:
            mne = scenario.get('mne', 5)
            mdi = scenario.get('mdi', 1)
        else:
            mne = scenario.get('mne', 5000)
            mdi = scenario.get('mdi', 5)
    else:
        mne = 0 # scenario.get('mne', 0)
        mdi = 0 # scenario.get('mdi', 0)
    # Fixed model layered.
    if scenario["model_fixed_layered"]:
        if scenario["small_model"]:
            mns = scenario.get('mns', 2)
            mds = scenario.get('mds', 3)
        else:
            mns = scenario.get('mns', 20)
            mds = scenario.get('mds', 64)
    else:
        mns = scenario.get('mns', 0)
        mds = scenario.get('mds', 0)
    # Aggregator only.
    if scenario["aggregator_only"]:
        mns = scenario.get('mns', 0)
        mds = scenario.get('mds', 0)
        mdo = scenario.get('mdo', 0)
    # Numeric output.
    if scenario["output_numeric"]:
        if scenario["small_data"]:
            ydn = scenario.get('ydn', 2)
        else:
            ydn = scenario.get('ydn', 20)
    else:
        ydn = scenario.get('ydn', 0)
    # Categorical output.
    if scenario["output_categorical"]:
        if scenario["small_data"]:
            yne = scenario.get('yne', 5)
            ydi = scenario.get('ydi', 1)
        else:
            yne = scenario.get('yne', 5000)
            ydi = scenario.get('ydi', 5)
    else:
        yne = scenario.get('yne', 0)
        ydi = scenario.get('ydi', 0)
    # Weighted output dimensioned.
    if scenario["weighted_output"]:
        if scenario["weights_dimensioned"]:
            ywd = scenario.get('ywd', ydn + yne)
        else:
            ywd = scenario.get('ywd', 1)
    else:
        ywd = scenario.get('ywd', 0)
    # Multithreading.
    if scenario["threaded"]:
        num_threads = scenario.get('num_threads', None)
    else:
        num_threads = scenario.get('num_threads', 1)
    # Generate the model config.
    config = AXY.new_model_config(
        adn=adn,
        ade=scenario.get("ade",None),
        ane=ane,
        ads=ads,
        ans=ans,
        ado=scenario.get("ado",None),
        mdn=mdn,
        mde=scenario.get("mde",None),
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
    import numpy as np
    np.random.seed(seed)
    # Generate data that matches these specifications.
    ftype = dict(order="F", dtype="float32")
    itype = dict(order="F", dtype="int32")
    asarray = lambda a, t: np.asarray(a, **t)
    ax_in = asarray(np.random.normal(size=(adn, config.nat)), ftype)
    axi_in = asarray(np.random.randint(*na_range, size=(adi, config.nat)), itype)
    sizes_in = asarray(np.random.randint(0, max(1,round(2*(config.nat / nm_in))), size=(nm_in if config.nat > 0 else 0,)), itype)
    x_in = asarray(np.random.normal(size=(mdn, nm_in)), ftype)
    xi_in = asarray(np.random.randint(*nm_range, size=(mdi, nm_in)), itype)
    y_in = asarray(np.random.normal(size=(ydn, nm_in)), ftype)
    if (ydi > 0): yi_in = asarray(np.random.randint(1, yne, size=(ydi, nm_in)), itype)
    else:         yi_in = None
    yw_in = asarray(np.random.normal(size=(ywd, nm_in)), ftype)
    # Set two of the sizes to zero (to make sure there are zeros in there.
    if (len(sizes_in) > 0):
        sizes_in[1*len(sizes_in) // 3] = 0
        sizes_in[2*len(sizes_in) // 3] = 0
    # Adjust the sizes to make sure the sum is the correct value.
    ssum = sum(sizes_in)
    i = 0
    while (ssum != config.nat):
        if (ssum < config.nat):
            ssum += 1
            sizes_in[i] += 1
        elif (ssum > config.nat) and (sizes_in[i] > 0):
            ssum -= 1
            sizes_in[i] -= 1
        i = (i + 1) % len(sizes_in)
    # Generate the memory and references to specific data holders.
    details = Details(config, scenario['steps'], ydi=ydi, ywd=ywd)
    # Initialize the model.
    AXY.init_model(config, details.model, seed=seed)
    # Data holders.
    ax = details.ax
    axi = details.axi
    sizes = details.sizes
    x = details.x
    xi = details.xi
    y = details.y
    yi = details.yi
    yw = details.yw
    # Return the config and data.
    return (
        config, details,
        dict(ax_in=ax_in, axi_in=axi_in, sizes_in=sizes_in, x_in=x_in, xi_in=xi_in, y_in=y_in, yi_in=yi_in, yw_in=yw_in),
        dict(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, y=y, yi=yi, yw=yw),
    )


# Define a scenario generator.
def scenario_generator(scenario=SCENARIO, randomized=True, seed=None):
    # If this is randomized, overwrite the "range" function with a random one.
    if randomized:
        if (seed is not None):
            import random
            random.seed(0)
        from tlux.random import random_range
        range = random_range
    else:
        import builtins
        range = builtins.range
    # Get all keys.
    keys = sorted((k for k,v in scenario.items() if (type(v) is bool)))
    # Iterate over all binary pairs of keys.
    for i in range(2**len(keys)):
        # Start with the default scenario.
        s = scenario.copy()
        # Decide which combination of keys to set to True (using binary digits)
        bin_str = bin(i)[2:][::-1] + '0'*len(keys)
        for v,n in zip(bin_str, keys):
            s[n] = (v == '1')
        # Skip any configurations that affect the model when it is absent.
        if (s["aggregator_only"] and (
                s["input_fixed_numeric"] or
                s["input_fixed_categorical"] or
                s["model_fixed_layered"]
        )):
            pass
        # Skip the invalid configuration where outputs are NOT weighted and
        #  "weights_dimensioned" is True, that is meaningless since there are no weights.
        elif ((not s["weighted_output"]) and (s["weights_dimensioned"])):
            pass
        # Skip the invalid configuration where there are no outputs.
        elif (not (s["output_numeric"] or s["output_categorical"])):
            pass
        # Skip the invalid configuration for "large model" and NOT "model layered"
        elif not (s["small_model"] or 
                  s["model_fixed_layered"] or
                  s["model_aggregate_layered"]):
            pass
        # Skip ss involving the aggregator when it is not present.
        elif ((not s["input_aggregate_numeric"]) and
              (not s["input_aggregate_categorical"]) and
              (s["batch_aggregate_constrained"] or s["model_aggregate_layered"] or s["aggregate_pairwise"])):
            pass
        # Skip ss involving the model when it is not present.
        elif ((not s["input_fixed_numeric"]) and
              (not s["input_fixed_categorical"]) and
              (s["batch_aggregate_constrained"] or s["model_aggregate_layered"])):
            pass
        else:
            yield s


# Initialize an aggregate iterator given a config and list of sizes.
def initialize_agg_iterator(config, agg_iterators, sizes_in):
    for i in range(agg_iterators.shape[1]):
        if (sizes_in[i] == 0):
            agg_iterators[:,i] = 0
        else:
            agg_iterators[0,i] = sizes_in[i]
            if config.pairwise_aggregation:
                agg_iterators[0,i] = agg_iterators[0,i]**2
            agg_iterators[1:,i] = AXY.initialize_iterator(
                i_limit=agg_iterators[0,i],
            )



