import os
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_source_code = os.path.join(_this_dir, "apos.f90")

class APOS:
    # Make the string function return the unpacked model.
    def __str__(self): return str(self.model_unpacked())


    # Initialize a new APOS model.
    def __init__(self, **kwargs):
        import fmodpy
        f_compiler_args = "-fPIC -shared -O3 -lblas -fopenmp"
        # f_compiler_args = "-fPIC -shared -O3 -lblas -fopenmp -fcheck=bounds"
        apos = fmodpy.fimport(_source_code, blas=True, omp=True, wrap=True,
                              verbose=False, output_dir=_this_dir,
                              f_compiler_args=f_compiler_args)
        # Store the Fortran module as an attribute.
        self.APOS = apos.apos
        # Set defaults for standard internal parameters.
        self.steps = 1000
        self.seed = None
        self.config = None
        self.model = np.zeros(0, dtype="float32")
        self.record = np.zeros(0, dtype="float32")
        # Default descriptors for categorical inputs.
        self.m_map = []
        self.m_sizes = []
        self.m_starts = []
        self.a_map = []
        self.a_sizes = []
        self.a_starts = []
        # Initialize the attributes of the model that can be initialized.
        self._init_model(**kwargs)


    # Initialize a model, if possible.
    def _init_model(self, **kwargs):
        # Model parameters.
        mdi = kwargs.pop("mdi", None)
        mdo = kwargs.pop("mdo", None)
        mds = kwargs.pop("mds", None)
        mns = kwargs.pop("mns", None)
        mde = kwargs.pop("mde", None)
        mne = kwargs.pop("mne", None)
        # Apositional model parameters.
        adi = kwargs.pop("adi", 0)
        ado = kwargs.pop("ado", None)
        ads = kwargs.pop("ads", None)
        ans = kwargs.pop("ans", None)
        ade = kwargs.pop("ade", None)
        ane = kwargs.pop("ane", None)
        # Number of threads.
        num_threads = kwargs.pop("num_threads", None)
        self.seed = kwargs.pop("seed", self.seed)
        self.steps = kwargs.pop("steps", self.steps)
        # Initialize if enough arguments were provided.
        if (None not in {mdi, mdo, adi}):
            self.config = self.APOS.new_model_config(
                mdi=mdi, mdo=mdo, mds=mds, mns=mns, mne=mne, mde=mde,
                adi=adi, ado=ado, ads=ads, ans=ans, ane=ane, ade=ade,
                num_threads=num_threads)
            # Set any configuration keyword arguments given at initialization
            #  that were not passed to "new_model_config".
            for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
                setattr(self.config, n, kwargs[n])
            # Set all internal arrays and initialize the model.
            self.model = np.zeros(self.config.total_size, dtype="float32")
            self.APOS.init_model(self.config, self.model, seed=self.seed)


    # Unpack the model (which is in one array) into it's constituent parts.
    def model_unpacked(self):
        # If there is no model or configuration, return None.
        if (self.config is None) or (self.model is None):
            return None
        # Build a class that contains pointers to the model internals.
        class ModelUnpacked:
            config = self.config
            model  = self.model
            m_embeddings   = self.model[self.config.msev-1:self.config.meev].reshape(self.config.mde, self.config.mne, order="F")
            m_input_vecs   = self.model[self.config.msiv-1:self.config.meiv].reshape(self.config.mdi, self.config.mds, order="F")
            m_input_shift  = self.model[self.config.msis-1:self.config.meis].reshape(self.config.mds, order="F")
            m_state_vecs   = self.model[self.config.mssv-1:self.config.mesv].reshape(self.config.mds, self.config.mds, max(0,self.config.mns-1), order="F")
            m_state_shift  = self.model[self.config.msss-1:self.config.mess].reshape(self.config.mds, self.config.mns-1, order="F")
            m_output_vecs  = self.model[self.config.msov-1:self.config.meov].reshape(self.config.mds, self.config.mdo, order="F")
            a_embeddings   = self.model[self.config.asev-1:self.config.aeev].reshape(self.config.ade, self.config.ane, order="F")
            a_input_vecs   = self.model[self.config.asiv-1:self.config.aeiv].reshape(self.config.adi, self.config.ads, order="F")
            a_input_shift  = self.model[self.config.asis-1:self.config.aeis].reshape(self.config.ads, order="F")
            a_state_vecs   = self.model[self.config.assv-1:self.config.aesv].reshape(self.config.ads, self.config.ads, max(0,self.config.ans-1), order="F")
            a_state_shift  = self.model[self.config.asss-1:self.config.aess].reshape(self.config.ads, max(0,self.config.ans-1), order="F")
            a_output_vecs  = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.ads, self.config.ado, order="F")

            def __getitem__(self, *args, **kwargs):
                return getattr(self, *args, **kwargs)
            def __str__(self, vecs=False):
                # Calculate the byte size of this model (excluding python descriptors).
                byte_size = len(self.config._fields_)*4 + self.model.dtype.itemsize*self.model.size
                if (byte_size < 2**10):
                    byte_size = f"{byte_size} bytes"
                elif (byte_size < 2**20):
                    byte_size = f"{byte_size//2**10:.1f}KB"
                elif (byte_size < 2**30):
                    byte_size = f"{byte_size//2**20:.1f}MB"
                else:
                    byte_size = f"{byte_size//2**30:.1f}GB"
                # Create a function that prints the actual contents of the arrays.
                if vecs: to_str = lambda arr: "\n    " + "\n    ".join(str(arr).split("\n")) + "\n"
                else:    to_str = lambda arr: "\n"
                # Provide details (and some values where possible).
                return (
                    f"APOS model ({self.config.total_size} parameters) [{byte_size}]\n"+
                     " apositional model\n"+
                    f"  input dimension  {self.config.adi}\n"+
                    f"  output dimension {self.config.ado}\n"+
                    f"  state dimension  {self.config.ads}\n"+
                    f"  number of states {self.config.ans}\n"+
                   (f"  embedding dimension  {self.config.ade}\n"+
                    f"  number of embeddings {self.config.ane}\n"
                     if self.config.ane > 0 else "")+
                    f"  embeddings   {self.a_embeddings.shape}  "+to_str(self.a_embeddings)+
                    f"  input vecs   {self.a_input_vecs.shape}  "+to_str(self.a_input_vecs)+
                    f"  input shift  {self.a_input_shift.shape} "+to_str(self.a_input_shift)+
                    f"  state vecs   {self.a_state_vecs.shape}  "+to_str(self.a_state_vecs)+
                    f"  state shift  {self.a_state_shift.shape} "+to_str(self.a_state_shift)+
                    f"  output vecs  {self.a_output_vecs.shape} "+to_str(self.a_output_vecs)+
                     "\n"+
                     " positional model\n"+
                    f"  input dimension  {self.config.mdi}\n"+
                    f"  output dimension {self.config.mdo}\n"+
                    f"  state dimension  {self.config.mds}\n"+
                    f"  number of states {self.config.mns}\n"+
                   (f"  embedding dimension  {self.config.mde}\n"+
                    f"  number of embeddings {self.config.mne}\n"
                     if self.config.mne > 0 else "")+
                    f"  embeddings   {self.m_embeddings.shape}  "+to_str(self.m_embeddings)+
                    f"  input vecs   {self.m_input_vecs.shape}  "+to_str(self.m_input_vecs)+
                    f"  input shift  {self.m_input_shift.shape} "+to_str(self.m_input_shift)+
                    f"  state vecs   {self.m_state_vecs.shape}  "+to_str(self.m_state_vecs)+
                    f"  state shift  {self.m_state_shift.shape} "+to_str(self.m_state_shift)+
                    f"  output vecs  {self.m_output_vecs.shape} "+to_str(self.m_output_vecs)
                )
        return ModelUnpacked()


    # Fit this model.
    def fit(self, x=None, y=None, xi=None, ax=None, axi=None,
            sizes=None, new_model=False, **kwargs):
        # Ensure that 'y' values were provided.
        assert (y is not None), "APOS.fit requires 'y' values, but not provided (use keyword argument 'y=<values>')."
        # Make sure that 'sizes' were provided for apositional (aggregate) inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "APOS.fit requires 'sizes' to be provided for apositional input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        if (y is not None): y = np.asarray(y, dtype="float32", order="C")
        mn = y.shape[0]
        if (x is not None): x = np.asarray(x, dtype="float32", order="C")
        else:               x = np.zeros((mn,0), dtype="float32", order="C")
        if (xi is not None): xi = np.asarray(xi)
        else:                xi = np.zeros((mn,0), dtype="int32", order="C")
        if (sizes is not None): sizes = np.asarray(sizes, dtype="int32")
        else:                   sizes = np.zeros(0, dtype="int32")
        an = sizes.sum()
        if (ax is not None): ax = np.asarray(ax, dtype="float32", order="C")
        else:                ax = np.zeros((an,0), dtype="float32", order="C")
        if (axi is not None): axi = np.asarray(axi)
        else:                 axi = np.zeros((an,0), dtype="int32", order="C")
        # Make sure that all inputs have the expected shape.
        assert (len(y.shape) in {1,2}), f"Bad y shape {y.shape}, should be 1D or 2D matrix."
        assert (len(x.shape) in {1,2}), f"Bad x shape {x.shape}, should be 1D or 2D matrix."
        assert (len(xi.shape) in {1,2}), f"Bad xi shape {xi.shape}, should be 1D or 2D matrix."
        assert (len(ax.shape) in {1,2}), f"Bad ax shape {ax.shape}, should be 1D or 2D matrix."
        assert (len(axi.shape) in {1,2}), f"Bad axi shape {axi.shape}, should be 1D or 2D matrix."
        assert (len(sizes.shape) == 1), f"Bad sizes shape {sizes.shape}, should be 1D int vectora."
        # Reshape inputs to all be two dimensional (except sizes).
        if (len(y.shape) == 1): y = y.reshape((-1,1))
        mdo = y.shape[1]
        if (len(x.shape) == 1): x = x.reshape((-1,1))
        mdi = x.shape[1]
        if (len(xi.shape) == 1): xi = xi.reshape((-1,1))
        if (len(ax.shape) == 1): ax = ax.reshape((-1,1))
        adi = ax.shape[1]
        if (len(axi.shape) == 1): axi = axi.reshape((-1,1))
        # TODO: Extract all of the code above ^^^ into a separate
        #       "_to_arrays" function. Find overlap with 'predict'.
        # TODO: Add support for 'yi', add y components for unique values
        #       and modify .predict to output the predicted unique value.
        # TODO: Extract below to "_unique_aggregation" function, remove
        #       redundancy between xi and axi.
        # Transform cetegorical inputs into expected format for model.
        if (xi.shape[1] > 0):
            self.m_map = [np.unique(xi[:,i]) for i in range(xi.shape[1])]
            self.m_sizes = [len(u) for u in self.m_map]
            self.m_starts = (np.cumsum(self.m_sizes) - self.m_sizes[0] + 1).tolist()
            mne = sum(self.m_sizes)
            _xi = np.zeros((mn, xi.shape[1]), dtype="int32", order="C")
            for i in range(xi.shape[1]):
                start_index = self.m_starts[i]
                num_unique = self.m_sizes[i]
                unique_vals = self.m_map[i]
                eq_val = xi[:,i].reshape(-1,1) == unique_vals
                val_indices = np.ones((mn,num_unique), dtype="int32") * (
                    start_index + np.arange(num_unique))
                _xi[:,i] = val_indices[eq_val]
            xi = _xi
        else: mne = 0
        if (axi.shape[1] > 0):
            self.a_map = [np.unique(axi[:,i]) for i in range(axi.shape[1])]
            self.a_sizes = [len(u) for u in self.a_map]
            self.a_starts = (np.cumsum(self.a_sizes) - self.a_sizes[0] + 1).tolist()
            ane = sum(self.a_sizes)
            _axi = np.zeros((an, axi.shape[1]), dtype="int32", order="C")
            for i in range(axi.shape[1]):
                start_index = self.a_starts[i]
                num_unique = self.a_sizes[i]
                unique_vals = self.a_map[i]
                eq_val = axi[:,i].reshape(-1,1) == unique_vals
                val_indices = np.ones((an,num_unique), dtype="int32") * (
                    start_index + np.arange(num_unique))
                _axi[:,i] = val_indices[eq_val]
            axi = _axi
        else: ane = 0
        # ------------------------------------------------------------
        # If the shape of the model does not match the data, reinitialize.
        check_shape = lambda: self.APOS.check_shape(self.config, self.model, y.T, x.T, xi.T, ax.T, axi.T, sizes)
        if (new_model or (self.config is None) or (check_shape() != 0)):
            # Store any existing configurations.
            if (self.config is not None):
                kwargs.update({
                    "mne": kwargs.get("mne",self.config.mne),
                    "mns": kwargs.get("mns",self.config.mns),
                    "mds": kwargs.get("mds",self.config.mds),
                    "ane": kwargs.get("ane",self.config.ane),
                    "ans": kwargs.get("ans",self.config.ans),
                    "ads": kwargs.get("ads",self.config.ads),
                    "ado": kwargs.get("ado",self.config.ado),
                })
                import warnings
                warnings.warn(f"Creating new model config because 'check_shape' failed. Only keeping sizes, dropping all other custom configurations.")
            # Ensure that the config is compatible with the data.
            kwargs.update({
                "mdi":mdi,
                "mdo":mdo,
                "adi":adi,
                "ane":max(ane, kwargs.get("ane",0)),
                "mne":max(mne, kwargs.get("mne",0)),
            })
            self._init_model(**kwargs)
        # If a random seed is provided, then only 2 threads can be used
        #  because nondeterministic behavior is exhibited otherwise.
        if (self.seed is not None):
            if (self.config.num_threads > 2):
                import warnings
                warnings.warn("Seeding an APOS model will deterministically initialize weights, but num_threads > 2 will result in a nondeterministic model fit.")
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # ------------------------------------------------------------
        # Minimize the mean squared error.
        self.record = np.zeros((steps,4), dtype="float32", order="C")
        result = self.APOS.minimize_mse(self.config, self.model,
                                        y.T, x.T, xi.T, ax.T, axi.T, sizes,
                                        steps=steps, record=self.record.T)
        assert (result[-1] == 0), f"APOS.minimize_mse returned nonzero exit code {result[-1]}."


    # Calling this model is an alias for 'APOS.predict'.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    # Make predictions for new data.
    def predict(self, x=None, xi=None, ax=None, axi=None, sizes=None,
                embedding=False, save_states=False, **kwargs):
        # Evaluate the model at all data.
        assert ((x is not None) or (sizes is not None)), "APOS.predict requires at least one of 'x' or 'sizes' to not be None."
        # Make sure that 'sizes' were provided for apositional (aggregate) inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "APOS.predict requires 'sizes' to be provided for apositional input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        if (x is not None):
            x = np.asarray(x, dtype="float32", order="C")
            mn = x.shape[0]
        if (sizes is not None):
            sizes = np.asarray(sizes, dtype="int32")
            mn = sizes.shape[0]
        if (x is None):     x = np.zeros((mn,0), dtype="float32", order="C")
        if (sizes is None): sizes = np.zeros(0, dtype="int32")
        an = sizes.sum()
        if (xi is not None): xi = np.asarray(xi)
        else:                xi = np.zeros((mn,0), dtype="int32", order="C")
        if (ax is not None): ax = np.asarray(ax, dtype="float32", order="C")
        else:                ax = np.zeros((an,0), dtype="float32", order="C")
        if (axi is not None): axi = np.asarray(axi)
        else:                 axi = np.zeros((an,0), dtype="int32", order="C")
        # Make sure that all inputs have the expected shape.
        assert (len(x.shape) in {1,2}), f"Bad x shape {x.shape}, should be 1D or 2D matrix."
        assert (len(xi.shape) in {1,2}), f"Bad xi shape {xi.shape}, should be 1D or 2D matrix."
        assert (len(ax.shape) in {1,2}), f"Bad ax shape {ax.shape}, should be 1D or 2D matrix."
        assert (len(axi.shape) in {1,2}), f"Bad axi shape {axi.shape}, should be 1D or 2D matrix."
        assert (len(sizes.shape) == 1), f"Bad sizes shape {sizes.shape}, should be 1D int vectora."
        # Reshape inputs to all be two dimensional (except sizes).
        if (len(x.shape) == 1): x = x.reshape((-1,1))
        mdi = x.shape[1]
        if (len(xi.shape) == 1): xi = xi.reshape((-1,1))
        if (len(ax.shape) == 1): ax = ax.reshape((-1,1))
        adi = ax.shape[1]
        if (len(axi.shape) == 1): axi = axi.reshape((-1,1))
        # Make sure the categorical inputs have the expected shape.
        assert (xi.shape[1] == len(self.m_map)), f"Bad xi shape {xi.shape}, expected {len(self.m_map)} columns."
        assert (axi.shape[1] == len(self.a_map)), f"Bad axi shape {axi.shape}, expected {len(self.a_map)} columns."
        # Transform cetegorical inputs into expected format for model.
        if (xi.shape[1] > 0):
            mne = self.m_starts[-1] - 1
            _xi = np.zeros((mn, xi.shape[1]), dtype="int32", order="C")
            for i in range(xi.shape[1]):
                start_index = self.m_starts[i]
                num_unique = self.m_sizes[i]
                unique_vals = self.m_map[i]
                eq_val = xi[:,i].reshape(-1,1) == unique_vals
                # Add a column to the front that is the default if none match.
                eq_val = np.concatenate((
                    np.logical_not(eq_val.max(axis=1)).reshape(mn,1),
                    eq_val), axis=1)
                val_indices = np.ones((mn,num_unique+1), dtype="int32") * np.arange(num_unique+1)
                val_indices[:,1:] += start_index-1
                _xi[:,i] = val_indices[eq_val]
            xi = _xi
        else: mne = 0
        if (axi.shape[1] > 0):
            ane = self.a_starts[-1] - 1
            _axi = np.zeros((an, axi.shape[1]), dtype="int32", order="C")
            for i in range(axi.shape[1]):
                start_index = self.a_starts[i]
                num_unique = self.a_sizes[i]
                unique_vals = self.a_map[i]
                eq_val = axi[:,i].reshape(-1,1) == unique_vals
                # Add a column to the front that is the default if none match.
                eq_val = np.concatenate((
                    np.logical_not(eq_val.max(axis=1)).reshape(an,1),
                    eq_val), axis=1)
                val_indices = np.ones((an,num_unique+1), dtype="int32") * np.arange(num_unique+1)
                val_indices[:,1:] += start_index-1
                _axi[:,i] = val_indices[eq_val]
            axi = _axi
        else: ane = 0
        # Embed the inpputs into the purely positional form.
        ade = self.config.ade
        ads = self.config.ads
        ado = self.config.ado
        mde = self.config.mde
        mds = self.config.mds
        mdo = self.config.mdo
        # Compute the true real-vector input dimensions given embeddings.
        adi += ade
        mdi += mde + ado
        # Initialize storage for all arrays needed at evaluation time.
        y = np.zeros((mn, mdo), dtype="float32", order="C")
        xxi = np.zeros((mn, mdi), dtype="float32", order="C")
        axxi = np.zeros((an, adi), dtype="float32", order="C")
        if (save_states):
            mns = self.config.mns
            m_states = np.zeros((mn, mds, mns), dtype="float32", order="F")
            ans = self.config.ans
            a_states = np.zeros((an, ads, ans), dtype="float32", order="F")
        else:
            m_states = np.zeros((mn, mds, 2), dtype="float32", order="F")
            a_states = np.zeros((an, ads, 2), dtype="float32", order="F")
        ay = np.zeros((an, ado), dtype="float32", order="F")
        # Call the unerlying library.
        info = self.APOS.check_shape(self.config, self.model, y.T, x.T, xi.T, ax.T, axi.T, sizes)
        assert (info == 0), f"APOS.predict encountered nonzero exit code {info} when calling APOS.check_shape."
        self.APOS.embed(self.config, self.model, x.T, xi.T, ax.T, axi.T, xxi.T, axxi.T)
        result = self.APOS.evaluate(self.config, self.model, y.T, xxi.T, axxi.T,
                                    sizes, m_states, a_states, ay, info, **kwargs)
        assert (result[-1] == 0), f"APOS.evaluate returned nonzero exit code {result[-1]}."
        if (save_states):
            self.m_states = m_states
            self.a_states = a_states
        if (embedding):
            return m_states[:,:,0]
        else:
            return y


    # Save this model to a path.
    def save(self, path):
        import json
        with open(path, "w") as f:
            # Get the config as a Python type.
            if (self.config is None): config = None
            else: config = {n:getattr(self.config, n) for (n,t) in self.config._fields_}
            # Write the JSON file with Python types.
            f.write(json.dumps({
                # Create a dictionary of the known python attributes.
                "config" : config,
                "model"  : self.model.tolist(),
                "record" : self.record.tolist(),
                "m_map"    : [l.tolist() for l in self.m_map],
                "m_sizes"  : self.m_sizes,
                "m_starts" : self.m_starts,
                "a_map"    : [l.tolist() for l in self.a_map],
                "a_sizes"  : self.a_sizes,
                "a_starts" : self.a_starts,
            }))


    # Load this model from a path (after having been saved).
    def load(self, path):
        # Read the file.
        import json
        with open(path, "r") as f:
            attrs = json.loads(f.read())
        # Load the attributes of the model.
        for key in attrs:
            value = attrs[key]
            if (key[-4:] == "_map"):
                value = [np.asarray(l) for l in value]
            elif (key[:2] in {"m_","a_"}):
                pass
            elif (type(value) is list): 
                value = np.asarray(value, dtype="float32")
            setattr(self, key, value)
        # Convert the the dictionary configuration into the correct type.
        if (type(self.config) is dict):
            self.config = self.APOS.MODEL_CONFIG(**self.config)
        # Return self in case an assignment was made.
        return self


if __name__ == "__main__":
    print("_"*70)
    print(" TESTING APOS MODULE")

    from tlux.plot import Plot
    from tlux.random import well_spaced_ball, well_spaced_box

    # TODO: test saving and loading with unique value maps
    # TODO: design concise test function that has meaningful signal
    #       in each of "ax", "axi", "x", "xi", test all combinations
    # TODO: make visualization optional for all of the tests

    # A function for testing approximation algorithms.
    def f(x):
        x = x.reshape((-1,2))
        x, y = x[:,0], x[:,1]
        return 3*x + np.cos(8*x)/2 + np.sin(5*y)
    seed = 1
    layer_dim = 32
    num_layers = 8
    steps = 10000
    num_threads = None
    np.random.seed(seed)


    TEST_SAVE_LOAD = False
    TEST_INT_INPUT = True
    TEST_APOSITIONAL = False
    TEST_VARIED_SIZE = False


    if TEST_SAVE_LOAD:
        # Try saving an untrained model.
        m = APOS()
        print("Empty model:")
        print("  str(model) =", str(m))
        print()
        m.save("testing_empty_save.json")
        m.load("testing_empty_save.json")
        from util.approximate import PLRM
        m = APOS(mdi=2, mds=layer_dim, mns=num_layers, mdo=1, seed=seed,
                 num_threads=num_threads, steps=steps, 
                 ) # discontinuity=-1000.0) # initial_step=0.01)
        print("Initialized model:")
        print(m)
        print()
        # Create the test plot.
        x = well_spaced_box(100, 2)
        y = f(x)
        # Normalize the data.
        x -= x.mean(axis=0)
        x /= x.var(axis=0)
        x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
        y -= y.mean(axis=0)
        y /= y.var(axis=0)
        # Fit the model.
        m.fit(x, y)
        # Add the data and the surface of the model to the plot.
        p = Plot()
        p.add("Data", *x.T, y)
        p.add_func("Fit", m, *x_min_max, vectorized=True)
        # Try saving the trained model and applying it after loading.
        print("Saving model:")
        print(m)
        print()
        m.save("testing_real_save.json")
        m.load("testing_real_save.json")
        print("Loaded model:")
        print(m)
        # print(str(m)[:str(m).index("\n\n")])
        print()
        p.add("Loaded values", *x.T, m(x)+0.05, color=1, marker_size=4)
        p.plot(show=(m.record.size == 0))
        # Add plot showing the training loss.
        if (m.record.size > 0):
            print("Generating loss plot..")
            p = type(p)("Mean squared error")
            # Rescale the columns of the record for visualization.
            record = m.record
            p.add("MSE", list(range(record.shape[0])), record[:,0], color=1, mode="lines")
            p.add("Step factors", list(range(record.shape[0])), record[:,1], color=2, mode="lines")
            p.add("Step sizes", list(range(record.shape[0])), record[:,2], color=3, mode="lines")
            p.add("Update ratio", list(range(record.shape[0])), record[:,3], color=4, mode="lines")
            p.show(append=True, show=True)
            print("", "done.", flush=True)
        # Remove the save files.
        import os
        try: os.remove("testing_empty_save.json")
        except: pass
        try: os.remove("testing_real_save.json")
        except: pass


    if TEST_INT_INPUT:
        print("Building model..")
        x = well_spaced_box(100, 2)
        x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
        y = f(x)
        # Initialize a new model.
        m = APOS(mdi=2, mds=layer_dim, mns=num_layers, mdo=1, mne=2, seed=seed, steps=steps, num_threads=num_threads)
        all_x = np.concatenate((x, x), axis=0)
        all_y = np.concatenate((y, np.cos(np.linalg.norm(x,axis=1))), axis=0)
        all_xi = np.concatenate((np.ones(len(x)),2*np.ones(len(x)))).reshape((-1,1)).astype("int32")
        m.fit(x=all_x, y=all_y, xi=all_xi)
        # Create an evaluation set that evaluates the model that was built over two differnt functions.
        xi1 = np.ones((len(x),1),dtype="int32")
        y1 = m(x, xi=xi1)
        y2 = m(x, xi=2*xi1)
        print("Adding to plot..")
        p = Plot()
        p.add("xi=1 true", *x.T, all_y[:len(all_y)//2], color=0)
        p.add("xi=2 true", *x.T, all_y[len(all_y)//2:], color=1)
        p.add_func("xi=1", lambda x: m(x, xi=np.ones(len(x), dtype="int32").reshape((-1,1))), [0,1], [0,1], vectorized=True, color=3, shade=True)
        p.add_func("xi=2", lambda x: m(x, xi=2*np.ones(len(x), dtype="int32").reshape((-1,1))), [0,1], [0,1], vectorized=True, color=2, shade=True)

        # Generate the visual.
        print("Generating surface plot..")
        p.show(show=False)
        print("Generating loss plot..")
        p = type(p)("Mean squared error")
        # Rescale the columns of the record for visualization.
        record = m.record
        p.add("MSE", list(range(record.shape[0])), record[:,0], color=1, mode="lines")
        p.add("Step factors", list(range(record.shape[0])), record[:,1], color=2, mode="lines")
        p.add("Step sizes", list(range(record.shape[0])), record[:,2], color=3, mode="lines")
        p.add("Update ratio", list(range(record.shape[0])), record[:,3], color=4, mode="lines")
        p.show(append=True, show=True)
        print("", "done.", flush=True)


    if TEST_APOSITIONAL:
        print("Building model..")
        x = well_spaced_box(100, 2)
        x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
        y = f(x)
        # Initialize a new model.
        m = APOS(adi=1, ane=2, mne=2, mdi=0, mdo=1, seed=seed, steps=steps, num_threads=num_threads)
        all_x = np.concatenate((x, x), axis=0)
        all_y = np.concatenate((y, np.cos(np.linalg.norm(x,axis=1))), axis=0)
        all_xi = np.concatenate((np.ones(len(x)),2*np.ones(len(x)))).reshape((-1,1)).astype("int32")
        ax = all_x.reshape((-1,1)).copy()
        axi = (np.ones(all_x.shape, dtype="int32") * (np.arange(all_x.shape[1])+1)).reshape(-1,1)
        sizes = np.ones(all_x.shape[0], dtype="int32") * 2
        temp_x = np.zeros((all_x.shape[0],0), dtype="float32")
        m.fit(x=temp_x, y=all_y, xi=all_xi, ax=ax, axi=axi, sizes=sizes, steps=1000)

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
        p.add("xi=1 true", *x.T, all_y[:len(all_y)//2], color=0)
        p.add("xi=2 true", *x.T, all_y[len(all_y)//2:], color=1)
        def fhat(x, i=1):
            xi = i * np.ones((len(x),1),dtype="int32")
            ax = x.reshape((-1,1)).copy()
            axi = (np.ones(x.shape, dtype="int32") * (np.arange(x.shape[1])+1)).reshape(-1,1)
            sizes = np.ones(x.shape[0], dtype="int32") * 2
            temp_x = np.zeros((x.shape[0],0), dtype="float32")
            return m(x=temp_x, xi=xi, ax=ax, axi=axi, sizes=sizes)
        p.add_func("xi=1", lambda x: fhat(x, 1), [0,1], [0,1], vectorized=True, color=3, mode="markers", shade=True)
        p.add_func("xi=2", lambda x: fhat(x, 2), [0,1], [0,1], vectorized=True, color=2, mode="markers", shade=True)
        # Generate the visual.
        print("Generating surface plot..")
        p.show(show=False)
        print("Generating loss plot..")
        p = type(p)("Mean squared error")
        # Rescale the columns of the record for visualization.
        record = m.record
        p.add("MSE", list(range(record.shape[0])), record[:,0], color=1, mode="lines")
        p.add("Step factors", list(range(record.shape[0])), record[:,1], color=2, mode="lines")
        p.add("Step sizes", list(range(record.shape[0])), record[:,2], color=3, mode="lines")
        p.add("Update ratio", list(range(record.shape[0])), record[:,3], color=4, mode="lines")
        p.show(append=True, show=True)
        print("", "done.", flush=True)


    if TEST_VARIED_SIZE:
        print("Creating data..")
        for test in range(100):
            print("sizes test: ", test, end="\r")
            sizes = np.random.randint(5,20,size=(10))
            an = sizes.sum()
            mn = sizes.size
            ax = np.random.random(size=(an,2))
            x = well_spaced_box(mn, 2)
            y = f(x)
            start = 0
            for i in range(len(sizes)):
                end = start + sizes[i]
                y[i] += ax[start:end].max()
                start = end
            # Fit a model.
            m = APOS(seed=seed, num_threads=num_threads, steps=1)
            m.fit(x=x, y=y, ax=ax, sizes=sizes)

