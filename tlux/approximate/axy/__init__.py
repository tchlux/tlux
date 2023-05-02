import os, re, math, sys, logging
import numpy as np
from tlux.approximate.axy.preprocessing import to_array
from tlux.unique import ByteArray

# TODO:
#  - create a 'plot_snapshot' method that creates a Graph of the model weights,
#    gradient magnitudes, curvatures, projected embeddings (colored by change),
#    and any other information that might be useful in diagnosing model components.
#  - draft solution to very slow data preparation for high dimensional "y",
#    either generate "well spaced" embeddings in lower dimension, or project.
#  - make "gradient" an accessible function that calls the model gradient code
#    and propogates an output gradient back to the inputs (+1 for all outputs).
#  - python fallback that supports the basic evaluation of
#    a model (but no support for training new models).

SHOW_FAILURE_PLOT = False

# Class for calling the underlying AXY model code.
class AXY:
    # Make the string function return the unpacked model.
    def __str__(self): return str(self.unpack())


    # Unpack the model (which is in one array) into it's constituent parts.
    def unpack(self, **kwargs):
        # If there is no model or configuration, return None.
        if (self.config is None) or (self.model is None):
            return None
        from tlux.approximate.axy.summary import AxyModel
        return AxyModel(self.config, self.model, clock_rate=self.AXY.clock_rate, **kwargs)


    # Initialize a new AXY model.
    def __init__(self, **kwargs):
        try:
            # Store the Fortran module as an attribute.
            from tlux.approximate.axy.axy import axy
            from tlux.approximate.axy.axy_random import random
            self.AXY = axy
            self.random = random
        except:
            from tlux.setup import build_axy
            try:
                axy, random = build_axy()
                self.AXY = axy.axy
                self.random = random.random
            except:
                raise(NotImplementedError("The Fortran source was not loaded successfully."))
        # Set defaults for standard internal parameters.
        self.steps = 1000
        self.seed = None
        self.num_threads = None
        self.config = None
        self.model = np.zeros(0, dtype="float32")
        self.record = np.zeros(0, dtype="float32")
        self.embedding_transform = np.zeros(0, dtype="float32")
        # Default descriptors for categorical inputs.
        self.axi_map = []
        self.xi_map = []
        self.yi_map = []
        self.yi_embeddings = np.zeros(0, dtype="float32")
        # Initialize the attributes of the model that can be initialized.
        self._init_kwargs = kwargs
        self._init_model(**kwargs)


    # Initialize a model, if possible.
    def _init_model(self, **kwargs):
        # Aggregator model parameters.
        adn = kwargs.pop("adn", 0)
        ado = kwargs.pop("ado", None)
        ads = kwargs.pop("ads", None)
        ans = kwargs.pop("ans", None)
        ade = kwargs.pop("ade", None)
        ane = kwargs.pop("ane", None)
        # Model parameters.
        mdn = kwargs.pop("mdn", None)
        mdo = kwargs.pop("mdo", None)
        mds = kwargs.pop("mds", None)
        mns = kwargs.pop("mns", None)
        mde = kwargs.pop("mde", None)
        mne = kwargs.pop("mne", None)
        # Number of threads.
        self.num_threads = kwargs.pop("num_threads", self.num_threads)
        self.seed = kwargs.pop("seed", self.seed)
        self.steps = kwargs.pop("steps", self.steps)
        # Initialize if enough arguments were provided.
        if (None not in {adn, mdn, mdo}):
            self.config = self.AXY.new_model_config(
                adn=adn, ado=ado, ads=ads, ans=ans, ane=ane, ade=ade,
                mdn=mdn, mdo=mdo, mds=mds, mns=mns, mne=mne, mde=mde,
                num_threads=self.num_threads)
            # Set any configuration keyword arguments given at initialization
            #  that were not passed to "new_model_config".
            for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
                setattr(self.config, n, kwargs[n])
            # Set all internal arrays and initialize the model.
            self.model = np.zeros(self.config.total_size, dtype="float32")
            initial_shift_range = kwargs.get("initial_shift_range", None)
            initial_output_scale = kwargs.get("initial_output_scale", None)
            self.AXY.init_model(self.config, self.model, seed=self.seed,
                                initial_shift_range=initial_shift_range,
                                initial_output_scale=initial_output_scale)


    # Given an exit code, check to make sure it is 0 (good), if not then read
    #  the source file to return a reason for the code if it can be found.
    def _check_code(self, exit_code, method, **info):
        if (exit_code != 0):
            # Read the reason from the Fortran source file.
            source = ""
            source_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axy.f90")
            if (os.path.exists(source_file)):
                with open(source_file, "r") as f:
                    source = f.read()
            reason = re.search(f"INFO = {exit_code}" + r"\s*![^\n]*\n", source)
            if (reason is not None):
                reason = reason.group().strip()
                reason = reason[reason.index("!")+1:].strip()
            else:
                reason = ""
            # Show the record if the crash was during a fit.
            if ((method == "fit_model") and (SHOW_FAILURE_PLOT)):
                print()
                from tlux.plot import Plot
                p = Plot("Model training record")
                # Rescale the columns of the record for visualization.
                record = self.record
                i = record.shape[0]
                step_indices = list(range(1,i+1))
                p.add("MSE", step_indices, record[:i,0], color=1, mode="lines")
                p.add("Step factors", step_indices, record[:i,1], color=2, mode="lines")
                p.add("Step sizes", step_indices, record[:i,2], color=3, mode="lines")
                p.add("Update ratio", step_indices, record[:i,3], color=4, mode="lines")
                p.add("Eval utilization", step_indices, record[:i,4], color=5, mode="lines")
                p.add("Grad utilization", step_indices, record[:i,5], color=6, mode="lines")
                p.show(show=False, y_range=[-.2, 1.2])
                from tlux.math import project
                from tlux.approximate.axy.summary import Details
                p = Plot("Data and weight projections")
                deets = Details(config=self.config, model=self.model, **info)
                for (attr,val) in sorted(deets.items()):
                    if ((attr in {"record", "model_grad", "agg_iterators", "a_order", "m_order", "a_lengths", "m_lengths"}) or
                        (not hasattr(val, "size")) or
                        (val.size <= 0) or
                        (not hasattr(val, "shape")) or
                        (len(val.shape) not in {2,3})
                    ): continue
                    # Handle possible matrices of data.
                    if (len(val.shape) == 2):
                        print(f" plotting '{attr}'.. {val.shape}")
                        if (attr == "ay"):
                            val = val.T
                        minval = np.nanmin(val)
                        maxval = np.nanmax(val)
                        p.add(attr+f" ({minval:.2e}, {maxval:.2e})",
                              *project(val[:1000,:1000].T, 2).T)
                    # Handle possible stacked matrices (state weights, state values).
                    elif (len(val.shape) == 3):
                        print(f" plotting '{attr}'..")
                        for i in range(val.shape[-1]):
                            print(f"   {i+1}.. {val[:,:,i].shape}")
                            minval = np.nanmin(val)
                            maxval = np.nanmax(val)
                            p.add(attr+f" {i+1} ({minval:.2e}, {maxval:.2e})",
                                  *project(val[:1000,:1000,i].T, 2).T)
                print()
                p.show(append=True, show=True)
            # Raise an assertion error.
            assert (exit_code == 0), f"AXY.{method} returned nonzero exit code {exit_code}. {reason}"


    # Fit this model given whichever types of data are available. *i are categorical, a* are aggregate. 
    def fit(self, ax=None, axi=None, sizes=None, x=None, xi=None,
            y=None, yi=None, yw=None, new_model=False, nm=None, na=None,
            callback=None, **kwargs):
        # Ensure that 'y' values were provided.
        assert ((y is not None) or (yi is not None)), "AXY.fit requires 'y' or 'yi' values, but neitherwere provided (use keyword argument 'y=<values>' or 'yi=<values>')."
        # Make sure that 'sizes' were provided for aggregate inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.fit requires 'sizes' to be provided for aggregated input sets (ax and axi)."
        # Get all inputs as arrays.
        nm_total, na_total, ax, axi, sizes, x, xi, y, yw_in, shapes, maps = (
            to_array(ax, axi, sizes, x, xi, y, yi, maps=dict(
                axi_map=self.axi_map, xi_map=self.xi_map,
                yi_map=self.yi_map, yi_embeddings=self.yi_embeddings,
            ))
        )
        for (k,v) in maps.items(): setattr(self, k, v)
        # TODO: Is this the correct retrieval of pairwise aggregation setting?
        pairwise = kwargs.get("pairwise_aggregation", False)
        pairwise = pairwise or self._init_kwargs.get("pairwise_aggregation", False)
        if (self.config is not None):
            pairwise = pairwise or self.config.pairwise_aggregation
        # Set the number of aggregate and model input limits.
        if (nm is None): nm = nm_total
        else:            nm = min(nm, nm_total)
        # Set the default NA to have space for all data.
        if (na is None):
            na = sum(sizes[np.argsort(sizes)[-nm:]] ** (2 if pairwise else 1))
        # Make sure NA is only as large as it practically needs to be.
        else:
            na = min(na, sum(sizes[np.argsort(sizes)[-nm:]] ** (2 if pairwise else 1)))
        # Configure this model if requested (or not already done).
        self._init_kwargs.update(kwargs)
        kwargs = self._init_kwargs
        if (new_model or (self.config is None)):
            # Remove "None" valued configs and presume their value should be inferred.
            if (('ado' in kwargs) and (kwargs['ado'] is None)):
                kwargs.pop("ado")
            if (('mdo' in kwargs) and (kwargs['mdo'] is None)):
                kwargs.pop("mdo")
            # Ensure that the config is compatible with the data.
            kwargs.update({
                "adn":shapes["adn"],
                "ane":max(shapes["ane"], kwargs.get("ane",0)),
                "ado":kwargs.get("ado", (shapes["mdo"] if (kwargs.get("mdo",None) == 0) else None)),
                "mdn":shapes["mdn"],
                "mne":max(shapes["mne"], kwargs.get("mne",0)),
                "mdo":kwargs.get("mdo", shapes["mdo"]),
            })
            self._init_model(**kwargs)
        else:
            # Set any configuration keyword arguments.
            for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
                if (kwargs[n] is not None):
                    setattr(self.config, n, kwargs[n])            
        # Make sure some silly user error didn't happen.
        if (x.shape[1] > 0):
            assert (self.config.mdo > 0), f"Found disabled model (mdo=0) with positive number of 'x' inputs. Either do not provide 'x', 'xi', or do not set 'mdo=0'."
        # ------------------------------------------------------------
        # If a random seed is provided, then only 2 threads can be used
        #  because nondeterministic behavior comes from reordered addition.
        if (self.seed is not None):
            if (self.config.num_threads > 2):
                import warnings
                warnings.warn("Seeding an AXY model will deterministically initialize weights, but num_threads > 2 will result in a nondeterministic model fit because changing addition order changes rounding behavior.")
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # ------------------------------------------------------------
        # Set up new work space for this minimization process.
        self.AXY.new_fit_config(
            nm=nm, na=na, nmt=nm_total, nat=na_total,
            adi=axi.shape[1], mdi=xi.shape[1],
            seed=self.seed, config=self.config
        )
        rwork = np.ones(self.config.rwork_size, dtype="float32")  # beware of allocation, heap vs stack
        rwork[:] = 0.0 # set to zeros because lots of gradients are stored there.
        iwork = np.ones(self.config.iwork_size, dtype="int32")
        lwork = np.ones(self.config.lwork_size, dtype="int64")
        agg_iterators = np.ones((nm_total, 5), dtype="int64")
        yw = np.ones((nm, yw_in.shape[1]), dtype="float32")
        self.record = np.zeros((steps,6), dtype="float32", order="C")
        # Check all shapes to validate.
        info = self.AXY.fit_check(self.config, self.model, rwork, iwork, lwork,
                                  ax.T, axi.T, sizes, x.T, xi.T, y.T, yw_in.T,
                                  yw.T, agg_iterators.T,
                                  steps=steps, record=self.record.T, sum_squared_error=0.0)
        # Check for a nonzero exit code.
        self._check_code(info, "fit_check")
        # Store the number of steps already taken.
        continuing = False
        target_steps = self.config.steps_taken + steps
        # Minimize the mean squared error.
        while (self.config.steps_taken < target_steps):
            result = self.AXY.fit_model(
                self.config, self.model, rwork, iwork, lwork,
                ax.T, axi.T, sizes, x.T, xi.T, y.T, yw_in.T,
                yw.T, agg_iterators.T,
                steps=steps, record=self.record.T,
                continuing=continuing,
            )
            # Check for a nonzero exit code.
            self._check_code(result[-1], "fit_model", steps=steps, rwork=rwork, iwork=iwork, lwork=lwork)
            # Update "continuing" to be true.
            continuing = True
            # If a callback was provided, call it (providing "Details" for the fit).
            if (callback is not None):
                from tlux.approximate.axy.summary import Details
                callback(
                    Details(
                        config=self.config, model=self.model,
                        steps=self.config.steps_taken,
                        rwork=rwork, iwork=iwork, lwork=lwork,
                        agg_iterators=agg_iterators, record=self.record,
                        yw=yw, yi=yi
                    )
                )
        # Store the multiplier to be used in embeddings (to level the norm contribution).
        if (self.config.mdo > 0):
            last_weights = self.model[self.config.msov-1:self.config.meov].reshape(self.config.mdso, self.config.mdo, order="F")
        else:
            last_weights = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.adso, self.config.ado+1, order="F")[:,:-1]
        self.embedding_transform = np.linalg.norm(last_weights, axis=1)
        # Normalize the embedding transformation to be unit norm.
        transform_norm = np.linalg.norm(self.embedding_transform)
        if (transform_norm > 0):
            self.embedding_transform /= transform_norm
        


    # Calling this model is an alias for 'AXY.predict'.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    # Make predictions for new data.
    def predict(self, x=None, xi=None, ax=None, axi=None, sizes=None,
                embedding=False, save_states=False, raw_scores=False, na=None, **kwargs):
        # Evaluate the model at all data.
        assert ((x is not None) or (xi is not None) or (sizes is not None)), "AXY.predict requires at least one of 'x', 'xi', or 'sizes' to not be None."
        assert (not (embedding and raw_scores)), "AXY.predict cannot provide both 'embedding=True' *and* 'raw_scores=True'."
        # Make sure that 'sizes' were provided for aggregate inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.predict requires 'sizes' to be provided for aggregated input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        nmt, nat, ax, axi, sizes, x, xi, _, _, _, _ = (
            to_array(ax, axi, sizes, x, xi, maps=dict(axi_map=self.axi_map, xi_map=self.xi_map))
        )
        # Set the default "number of aggregate inputs" allowed.
        if (na is None):
            if (self.config.pairwise_aggregation):
                na = sum(sizes**2)
            else:
                na = nat
        # Initialize holder for y output.
        y = np.zeros((nmt, self.config.do), dtype="float32", order="C")
        # ------------------------------------------------------------
        # Call the unerlying library to make sure input shapes are appropriate.
        info = self.AXY.check_shape(self.config, self.model, ax.T, axi.T, sizes, x.T, xi.T, y.T)
        # Check for a nonzero exit code.
        self._check_code(info, "check_shape")
        # Normalize the numeric inputs.
        _, _, info = self.AXY.normalize_inputs(self.config, self.model, ax.T, sizes, x.T)
        self._check_code(info, "normalize_inputs")
        # ------------------------------------------------------------
        # Handle pairwise aggregation (requires more scratch space and fetching pairs).
        if ((self.config.pairwise_aggregation) and (nat > 0)):
            # Initialize the config iterator to make it linearly iterate over the data..
            temp_confg = dict(
                nm = self.config.nm,
                nmt = self.config.nmt,
                na = self.config.na,
                nat = self.config.nat,
                i_next = self.config.i_next,
                i_step = self.config.i_step,
                i_mult = self.config.i_mult,
                i_mod = self.config.i_mod,
            )
            self.config.nm = nmt
            self.config.nmt = nmt
            self.config.i_next = 0
            self.config.i_step = 1
            self.config.i_mult = 1
            self.config.i_mod = nmt
            # Initialize all aggregate input iterators.
            fits_all = na >= sum(sizes**2)
            agg_iterators = np.ones((5, nmt), dtype="int64", order="F")
            for i in range(agg_iterators.shape[1]):
                if (sizes[i] == 0):
                    agg_iterators[:,i] = 0
                else:
                    agg_iterators[0,i] = sizes[i]
                    if self.config.pairwise_aggregation:
                        agg_iterators[0,i] = agg_iterators[0,i]**2
                    # If all fit, use a deterministic iterators [0, 1, 2, ...]
                    if fits_all:
                        agg_iterators[1:,i] = (0, 1, 1, agg_iterators[0,i]) # next, step, mult, mod
                    # Otherwise, use a random linear iterator.
                    else:
                        # TODO: This line of code will break since INITIALIZE_ITERATOR
                        #       now lives in the RANDOM module.
                        agg_iterators[1:,i] = self.random.initialize_iterator(
                            i_limit=agg_iterators[0,i],
                        )
            # Create scratch space for all data holders.
            self.config.na = na
            # Save the current values as "*_in" to separate them from the scratch space.
            ax_in = ax.T ; axi_in = axi.T ; sizes_in = sizes ; x_in = x.T ; xi_in = xi.T
            # Create scratch space.
            ax = np.zeros((self.config.adi, na), dtype="float32", order="F")
            axi = np.zeros((axi_in.shape[0], na), dtype="int64", order="F")
            sizes = np.zeros(nmt, dtype="int64", order="F")
            x = np.zeros((self.config.mdi, nmt), dtype="float32", order="F")
            xi = np.zeros((xi_in.shape[0], nmt), dtype="int64", order="F")
            y_in = yw_in = np.zeros((0,nmt), dtype="float32", order="F")
            # Use FETCH_DATA to do the pairwise aggreagtion.
            config, agg_iterators, ax, axi, sizes, x, xi, y_in, yw_in, na = self.AXY.fetch_data(
                config = self.config,
                agg_iterators = agg_iterators,
                ax_in = ax_in,
                ax = ax,
                axi_in = axi_in,
                axi = axi,
                sizes_in = sizes_in,
                sizes = sizes,
                x_in = x_in,
                x = x,
                xi_in = xi_in,
                xi = xi,
                y_in = y_in,
                y = y_in,
                yw_in = yw_in,
                yw = yw_in,
            )
            # Reset the config.
            for (name, value) in temp_confg.items():
                setattr(self.config, name, value)
            # Delete all scratch space.
            del ax_in, axi_in, sizes_in, x_in, xi_in, y_in, yw_in, agg_iterators, name, value, temp_confg
            ax = ax[:,:na].T
            axi = axi[:,:na].T
            ay = np.zeros((na, config.ado+1), dtype="float32", order="F")
            x = x.T
            xi = xi.T
        else:
            # ------------------------------------------------------------
            # 
            # Initialize storage for all arrays needed at evaluation time.
            #   If there are integer embeddings, expand "ax" and "x" to have
            #   space to hold those embeddings.
            if (self.config.ade > 0):
                _ax = np.zeros((na, self.config.adi), dtype="float32", order="C")
                _ax[:,:self.config.adn] = ax
                ax = _ax
            ay = np.zeros((na, self.config.ado+1), dtype="float32", order="F")
            if (self.config.mdo > 0) and ((self.config.mde > 0) or (self.config.ado > 0)):
                _x = np.zeros((nmt, self.config.mdi), dtype="float32", order="C")
                _x[:,:self.config.mdn] = x
                x = _x
        # If all internal states are being saved, the make more space, otherwise only two copies are needed.
        if (save_states):
            a_states = np.zeros((na, self.config.ads, self.config.ans), dtype="float32", order="F")
            m_states = np.zeros((nmt, self.config.mds, self.config.mns), dtype="float32", order="F")
        else:
            a_states = np.zeros((na, self.config.ads, 2), dtype="float32", order="F")
            m_states = np.zeros((nmt, self.config.mds, 2), dtype="float32", order="F")
        # ------------------------------------------------------------
        # Embed the categorical inputs as numeical inputs.
        self.AXY.embed(self.config, self.model, axi.T, xi.T, ax.T, x.T)
        # Evaluate the model.
        result = self.AXY.evaluate(self.config, self.model, ax.T, ay, sizes,
                                   x.T, y.T, a_states, m_states, info)
        self._check_code(result[-1], "evaluate")
        # Save the states if that's requested.
        if (save_states):
            self.states = dict(
                ax = ax,
                a_states = a_states,
                ay = ay,
                x = x,
                m_states = m_states
            )
        else:
            self.states = None
        # If embeddings are desired, multiply the last state by the 2-norm
        #  of output weights for each component of that last embedding.
        if (embedding):
            # Store the "last_state" representation of data before output,
            #  as well as the "last_weights" that preceed output.
            if (self.config.mdo > 0):
                if (self.config.mns > 0):
                    if (m_states.shape[-1] < self.config.mns):
                        last_state = m_states[:,:,-2]
                    else:
                        last_state = m_states[:,:,-1]
                else:
                    last_state = x
            else:  # There is only an aggregator, no model follows.
                if (self.config.ans > 0):
                    if (a_states.shape[-1] < self.config.ans):
                        last_state = a_states[:,:,-2]
                    else:
                        last_state = a_states[:,:,-1]
                    state = a_states[:,:,-1]
                else:
                    state = ax
                # Collapse the aggregate embeddings into their mean.
                last_state = np.zeros((nmt, state.shape[1]), dtype="float32")
                set_start = set_end = 0
                for i in range(sizes.shape[0]):
                    set_end += sizes[i]
                    last_state[i,:] = state[set_start:set_end].mean(axis=0)
                    set_start = set_end
            # Rescale the last state to linearly level according to outputs.
            if (self.embedding_transform.size > 0):
                last_state = last_state * self.embedding_transform
            # Return the last state.
            return last_state
        # If there are categorical outputs, then select by taking the max magnitude output.
        elif ((len(self.yi_map) > 0) and (not raw_scores)):
            # Get the number of category outputs and the number of numeric outputs.
            yne = sum(map(len, self.yi_map))
            ynn = y.shape[1]-(yne-1)
            # Get the numerical columns first.
            _y = [y[:,i] for i in range(ynn)]
            # Project the categorical embedding outputs into a matrix with "yne" columns.
            y = np.matmul(y[:,ynn:], self.yi_embeddings.T)
            # Iterate over each categorical output column, picking the category whose embedding had the largest value.
            start = 0
            for i in range(len(self.yi_map)):
                size = len(self.yi_map[i])
                _y.append(
                    [self.yi_map[i][j] for j in np.argmax(y[:,start:start+size], axis=1)]
                )
                start += size
            return np.asarray(_y, dtype=object).T
        # Otherwise simply return the numeric predicted outputs by the model.
        else:
            return y


    # Save this model to a path.
    def save(self, path, **encoding_kwargs):
        import json
        # Get the config as a Python type.
        if (self.config is None):
            config = None
        else:
            config = {n:getattr(self.config, n) for (n,t) in self.config._fields_}
        # Open the output file and write.
        json_model = json.dumps({
            # Create a dictionary of the known python attributes.
            "config" : config,
            "model"  : self.model.tolist(),
            "record" : self.record.tolist(),
            "embedding_transform" : self.embedding_transform.tolist(),
            "xi_map" : [m.tolist() for m in self.xi_map],
            "axi_map" : [m.tolist() for m in self.axi_map],
            "yi_map" : [m.tolist() for m in self.yi_map],
            "yi_embeddings" : self.yi_embeddings.tolist(),
        })
        # Write the JSON contents of the model to file.
        if (path.endswith(".gz")):
            import gzip
            with gzip.open(path, "wb") as f:
                f.write(json_model.encode(**encoding_kwargs))
        else:
            with open(path, "w") as f:
                f.write(json_model)


    # Load this model from a path (after having been saved).
    def load(self, path):
        # Read the file.
        import builtins, json
        # Handle compressed files or regular files.
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "rb") as f:
                attrs = json.loads(f.read().decode())
        else:
            with open(path, "r") as f:
                attrs = json.loads(f.read())
        # Load the attributes of the model.
        for key, value in attrs.items():
            # Convert the categorical map keys to the appropriate type.
            if (key.endswith("_map")):
                value = list(map(ByteArray.fromlist, value))
            # Otherwise all lists are actually numpy arrays (weights).
            elif (type(value) is list): 
                value = np.asarray(value, dtype="float32")
            setattr(self, key, value)
        # Convert the the dictionary configuration into the correct type.
        if (type(self.config) is dict):
            self.config = self.AXY.MODEL_CONFIG(**self.config)
        # Return self in case an assignment was made.
        return self


    # Generate a plot of the fit process.
    def plot(self, p=None, fit=True, append=True, show=True):
        if (p is None):
            from tlux.plot import Plot
            p = Plot("Mean squared error")
        # Plot the fit process.
        if fit:
            record = self.record
            for i in range(0, record.shape[0]+1, max(1,record.shape[0] // 100)):
                step_indices = list(range(1,i+1))
                p.add("MSE", step_indices, record[:i,0], color=1, mode="lines", frame=i)
                p.add("Step factors", step_indices, record[:i,1], color=2, mode="lines", frame=i)
                p.add("Step sizes", step_indices, record[:i,2], color=3, mode="lines", frame=i)
                p.add("Update ratio", step_indices, record[:i,3], color=4, mode="lines", frame=i)
                p.add("Eval utilization", step_indices, record[:i,4], color=5, mode="lines", frame=i)
                p.add("Grad utilization", step_indices, record[:i,5], color=6, mode="lines", frame=i)
            p.show(append=append, show=show, y_range=[-.2, 1.2])
        # TODO: Plot the internal vectors.
        # TODO: Add ability to use different plotting functions.
        return p


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000) #, threshold=sys.max something?)
    print("_"*70)
    print(" TESTING AXY MODULE")

    # Compile and import testing version of the modules (with testing related compilation flags).
    import fmodpy
    RANDOM_MOD = fmodpy.fimport(
        input_fortran_file = "axy_random.f90",
        name = "axy_random",
        wrap = True,
        # rebuild = True,
        verbose = False,
        f_compiler_args = "-fPIC -shared -O0 -pedantic -fcheck=bounds -ftrapv -ffpe-trap=invalid,overflow,underflow,zero",
    ).random
    AXY_MOD = fmodpy.fimport(
        input_fortran_file = "axy.f90",
        dependencies = ["axy_random.f90", "axy_matrix_operations.f90", "axy_sort_and_select.f90", "axy.f90"],
        name = "axy",
        blas = True,
        lapack = True,
        omp = True,
        wrap = True,
        # rebuild = True,
        verbose = True,
        f_compiler_args = "-fPIC -shared -O0 -pedantic -fcheck=bounds -ftrapv -ffpe-trap=invalid,overflow,underflow,zero",
    ).axy

    # Import codes that will be used for testing.
    from tlux.plot import Plot
    from tlux.random import well_spaced_box

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

    functions = [f1, f2, f3]

    seed = 0
    np.random.seed(seed)

    d = 2
    n = 2**7
    nm = (len(functions) * n) # // 3
    new_model = True
    use_a = True
    use_x = False
    use_y = True
    use_yi = True and (len(functions) == 1)
    use_nearest_neighbor = False

    # ONLY_SGD = dict(
    #     faster_rate = 1.0,
    #     slower_rate = 1.0,
    #     update_ratio_step = 0.0,
    #     step_factor = 0.001,
    #     step_mean_change = 0.1,
    #     step_curv_change = 0.00,
    #     # initial_curv_estimate = 1.0,
    #     basis_replacement = False,
    # )

    settings = dict(
        seed=seed,
        ads = 64,
        ans = 2,
        ado = None,
        mds = 64,
        mns = 2,
        # mdo = 0,  # Set to 0 to force only an aggregate model (no interaction between aggregates).
        # steps = 10000,
        # initial_curv_estimate = 1.0,
        # step_factor = 0.005,
        # faster_rate = 1.01,
        # slower_rate = 0.998,
        # min_update_ratio = 1.0,
        # max_step_factor = 0.005,
        # min_step_factor = 0.001,
        # num_threads = 20,
        # granular_parallelism = True,
        log_grad_norm_frequency = 1,
        rank_check_frequency = 10,
        early_stop = False,
        # ax_normalized = True,
        # ay_normalized = True,
        # x_normalized = True,
        # y_normalized = True,
        # pairwise_aggregation = True,
        # keep_best = False,
        # **ONLY_SGD
    )

    # 11 second fit  granular = False
    # 14 second fit  granular = True

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
    sizes = np.ones(n*len(functions), dtype="int64") * d

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

    # TOOD: Replace "yii" in "fhat" by using the following.
    # # This is simpler, this way the fhat function doesn't need to do complicated dictionary mapping.
    # for i in range(yi.shape[0]):
    #     for j in range(yi.shape[1]):
    #         yi[i,j] = yi_values[j][yi[i,j]]

    # 
    # import json
    # print()
    # print(json.dumps(dict(
    #     x=[[round(v,3) for v in row] for row in x.tolist()],
    #     y=[[round(v,3) for v in row] for row in y.tolist()],
    #     fx=[[0.633, 0.544]],
    # )))
    # print()
    # print()
    # exit()

    print()
    print("ax.shape:    ", ax.shape)
    print("axi.shape:   ", axi.shape)
    print("sizes.shape: ", sizes.shape)
    print("x.shape:     ", x.shape)
    print("xi.shape:    ", xi.shape)
    print("y.shape:     ", y.shape)
    print("yi.shape:    ", yi.shape)
    print()


    # Train a new model or load an existing one.
    path = 'temp-model.json'
    if (new_model or (not os.path.exists(path))):
        # Initialize a new model.
        print("Fitting model..")
        m = AXY(
            **settings,
        )
        # def show_status(details):
        #     print("status: ", details.steps)
        m.fit(
            ax=(ax.copy() if use_a else None),
            axi=(axi if use_a else None),
            sizes=(sizes if use_a else None),
            x=(x.copy() if use_x else None),
            xi=(xi if use_x else None),
            y=(y.copy() if use_y else None),
            yi=(yi if use_yi else None),
            nm = nm,
            # callback=show_status,
        )
        exit()
        # Save and load the model.
        print("  saving..", flush=True)
        m.save(path)

    print("Done fitting, loading..", flush=True)
    # Load the saved model.
    m = AXY()
    m.load(path)
    # Remove the saved model if only new models are desired.
    if new_model:
        os.remove(path)

    # Print the model.
    print()
    print(m, m.config)
    print()

    # Evaluate the model and compare with the data provided for training.
    # TODO: Add some evaluations of the categorical outputs.
    # fy = m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)
    # yy = np.concatenate((y.astype(object), yi.astype(object)), axis=1)

    # Build a nearest neighbor tree over the embeddings if preset.
    if (use_nearest_neighbor):
        from tlux.approximate.balltree import BallTree
        embeddings = m(
            ax=(ax.copy() if use_a else None),
            axi=(axi if use_a else None),
            sizes=(sizes if use_a else None),
            x=(x.copy() if use_x else None),
            xi=(xi if use_x else None),
            embedding=True
        )
        print("embeddings.shape: ", embeddings.shape)
        tree = BallTree(embeddings, build=False)

        print("Tree:")
        print(tree)


    # Define a function that evaluates the model with some different presets.
    def fhat(x, f=functions[0].__name__, yii=None):
        x = np.asarray(x, dtype="float32").reshape((-1,d)).copy()
        n = x.shape[0]
        xi = np.asarray([f]*n, dtype=object).reshape((-1,1))
        ax = x.reshape((-1,1)).copy()
        axi = np.asarray(list(zip(
            [f for _ in range(n*d)],
            [i+1 for _ in range(n) for i in range(d)]
        )), dtype=object)
        sizes = np.ones(x.shape[0], dtype="int64") * d
        if (not use_a): ax = axi = sizes = None
        if (not use_x): x = xi = None
        if (not use_yi): yii = None
        if (not use_nearest_neighbor):
            if (yii is None): return m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, save_states=True)[:,0]
            else:             return [yi_values[yii][v] for v in m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)[:,(y.shape[1] if use_y else 0)+yii]]
        else:
            # Use the tree to lookup the nearest neighbor.
            emb = m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, embedding=True, save_states=True)
            i = tree(emb, return_distance=False)
            if (yii is None): return y[i,0]
            else:             return [yi_values[yii][v] for v in yi[i.flatten(),yii]]


    # Generate the plot of the surface for the predictions.
    print("Adding to plot..")
    p = Plot()
    plot_points = 3000
    # Add the two functions that are being approximated.
    if (use_y):
        # Add the provided points.
        for i, (f, xf) in enumerate(zip(functions, points)):
            p.add(f"xi={f.__name__}", *xf.T, f(xf), color=i, group=i)

        # Add the function approximations.
        for i,f in enumerate(functions):
            f = f.__name__
            p.add_func(f"xi={f}", lambda x: fhat(x, f=f), *x_min_max,
                       group=i,  vectorized=True, color=i, opacity=0.8,
                       plot_points=plot_points) # , mode="markers", marker_size=2)
    if (use_yi):
        # Add the provided points.
        for i, (f, xf) in enumerate(zip(functions, points)):
            for j in range(2):
                p.add(f"xi={f.__name__} yi={j}", *xf.T, [yi_values[j][v] for v in yi[:,j]],
                      color=j, group=f'c{j}', shade=True, marker_size=5)
        # Add the function approximations.
        for f in functions:
            for i in range(yi.shape[1]):
                p.add_func(f"xi={f.__name__}, fyi={i}", lambda x: fhat(x, f=f.__name__, yii=i),
                           *x_min_max, opacity=0.8, plot_points=plot_points,
                           mode="markers", shade=True, marker_size=4)
    p.plot(show=False)

    # Generate a visual for data projections.
    m.predict(
        ax=(ax.copy() if use_a else None),
        axi=(axi if use_a else None),
        sizes=(sizes if use_a else None),
        x=(x.copy() if use_x else None),
        xi=(xi if use_x else None),
        save_states=True
    )
    from tlux.math import project
    p = Plot("Data embeddings")
    munpacked = m.unpack()
    if (munpacked.a_embeddings.size > 0):
        p.add("a-embs", *project(munpacked.a_embeddings.T, 3).T)
    if (use_a):
        p.add("ax", *project(m.states["ax"], 3).T)
        p.add("ay", *project(m.states["ay"], 3).T)
    if (munpacked.m_embeddings.size > 0):
        p.add("m-embs", *project(munpacked.m_embeddings.T, 3).T)
    if (m.states["x"].size > 0):
        p.add("x", *project(m.states["x"], 3).T)

    # Generate a visual for the training loss.
    print("Generating surface plot..")
    # Generate a visual of the loss function.
    if (len(getattr(globals().get("m",None), "record", [])) > 0):
        p.show(append=True, show=False)
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
        p.show(append=True, show=True)
    print("", "done.", flush=True)
