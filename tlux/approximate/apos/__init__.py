import os
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_source_code = os.path.join(_this_dir, "apos.f90")

# Build a class that contains pointers to the model internals, allowing
#  python attribute access to all of the different components of the models.
class AposModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.a_embeddings  = self.model[self.config.asev-1:self.config.aeev].reshape(self.config.ade, self.config.ane, order="F")
        self.a_input_vecs  = self.model[self.config.asiv-1:self.config.aeiv].reshape(self.config.adi, self.config.ads, order="F")
        self.a_input_shift = self.model[self.config.asis-1:self.config.aeis].reshape(self.config.ads, order="F")
        self.a_state_vecs  = self.model[self.config.assv-1:self.config.aesv].reshape(self.config.ads, self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_state_shift = self.model[self.config.asss-1:self.config.aess].reshape(self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_output_vecs = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.adso, self.config.ado, order="F")
        self.m_embeddings  = self.model[self.config.msev-1:self.config.meev].reshape(self.config.mde, self.config.mne, order="F")
        self.m_input_vecs  = self.model[self.config.msiv-1:self.config.meiv].reshape(self.config.mdi, self.config.mds, order="F")
        self.m_input_shift = self.model[self.config.msis-1:self.config.meis].reshape(self.config.mds, order="F")
        self.m_state_vecs  = self.model[self.config.mssv-1:self.config.mesv].reshape(self.config.mds, self.config.mds, max(0,self.config.mns-1), order="F")
        self.m_state_shift = self.model[self.config.msss-1:self.config.mess].reshape(self.config.mds, max(0,self.config.mns-1), order="F")
        self.m_output_vecs = self.model[self.config.msov-1:self.config.meov].reshape(self.config.mdso, self.config.mdo, order="F")
        self.ax_shift = self.model[self.config.aiss-1:self.config.aise]
        self.ay_shift = self.model[self.config.aoss-1:self.config.aose]
        self.x_shift = self.model[self.config.miss-1:self.config.mise]
        self.y_shift = self.model[self.config.moss-1:self.config.mose]

    # Allow square brackets to access attributes of this model and its configuration.
    def __getitem__(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        elif hasattr(self.config, attr):
            return getattr(self.config, attr)

    # Create a summary string for this model.
    def __str__(self, vecs=False):
        # A function for creating a byte-size string from an integer.
        def _byte_str(byte_size):
            if (byte_size < 2**10):
                byte_size = f"{byte_size} bytes"
            elif (byte_size < 2**20):
                byte_size = f"{byte_size//2**10:.1f}KB"
            elif (byte_size < 2**30):
                byte_size = f"{byte_size//2**20:.1f}MB"
            elif (byte_size < 2**40):
                byte_size = f"{byte_size//2**30:.1f}GB"
            else:
                byte_size = f"{byte_size//2**40:.1f}TB"
            return byte_size
        # Calculate the byte size of this model (excluding python descriptors).
        # TODO: Not all configs are 4 bytes, do more expensive sum over actual sizes?
        byte_size = len(self.config._fields_)*4 + self.model.dtype.itemsize*self.model.size
        byte_size = _byte_str(byte_size)
        if (self.config.rwork_size+self.config.iwork_size > 0):
            work_size = self.config.rwork_size*4 + self.config.iwork_size*4
            byte_size += " + "+_byte_str(work_size)+" work space"
        # Create a function that prints the actual contents of the arrays.
        if vecs: to_str = lambda arr: "\n    " + "\n    ".join(str(arr).split("\n")) + "\n"
        else:    to_str = lambda arr: "\n"
        # Provide details (and some values where possible).
        return (
            f"APOS model ({self.config.total_size} parameters) [{byte_size}]\n"+
            (" apositional\n"+
            f"  input dimension  {self.config.adn}\n"+
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
             "\n" if (self.a_output_vecs.size > 0) else "") +
            (" positional\n"+
            f"  input dimension  {self.config.mdn}\n"+
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
             if (self.m_output_vecs.size > 0) else "")
        )


# Class for calling the underlying APOS model code.
class APOS:
    # Make the string function return the unpacked model.
    def __str__(self): return str(self.unpack())

    # Initialize a new APOS model.
    def __init__(self, **kwargs):
        try:
            import fmodpy
            apos = fmodpy.fimport(_source_code, blas=True,
                                  lapack=True, omp=True, wrap=True,
                                  verbose=False, output_dir=_this_dir,
            )
            # Store the Fortran module as an attribute.
            self.APOS = apos.apos
        except:
            # TODO:
            #  - python fallback that supports the basic evaluation of
            #    a model (but no support for training new models).
            raise(NotImplementedError("The Fortran source was not loaded successfully."))
        # Set defaults for standard internal parameters.
        self.steps = 1000
        self.seed = None
        self.num_threads = None
        self.config = None
        self.model = np.zeros(0, dtype="float32")
        self.record = np.zeros(0, dtype="float32")
        # Default descriptors for categorical inputs.
        self.axi_map = []
        self.axi_sizes = []
        self.axi_starts = []
        self.xi_map = []
        self.xi_sizes = []
        self.xi_starts = []
        self.yi_map = []
        self.yi_sizes = []
        self.yi_starts = []
        # Initialize the attributes of the model that can be initialized.
        self._init_model(**kwargs)


    # Initialize a model, if possible.
    def _init_model(self, **kwargs):
        # Apositional model parameters.
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
            self.config = self.APOS.new_model_config(
                adn=adn, ado=ado, ads=ads, ans=ans, ane=ane, ade=ade,
                mdn=mdn, mdo=mdo, mds=mds, mns=mns, mne=mne, mde=mde,
                num_threads=self.num_threads)
            # Set any configuration keyword arguments given at initialization
            #  that were not passed to "new_model_config".
            for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
                setattr(self.config, n, kwargs[n])
            # Set all internal arrays and initialize the model.
            self.model = np.zeros(self.config.total_size, dtype="float32")
            self.APOS.init_model(self.config, self.model, seed=self.seed)


    # Generate the string containing all the configuration information for this model.
    def config_str(self):
        s = ""
        max_n_len = max(map(len,(n for (n,t) in self.config._fields_)))
        max_t_len = max(map(len,(str(t).split("'")[1].split('.')[1]
                                 for (n,t) in self.config._fields_)))
        for (n,t) in self.config._fields_:
            t = str(t).split("'")[1].split('.')[1]
            s += f"  {str(t):{max_t_len}s}  {n:{max_n_len}s}  =  {getattr(self.config,n)}\n"
        return s


    # Unpack the model (which is in one array) into it's constituent parts.
    def unpack(self):
        # If there is no model or configuration, return None.
        if (self.config is None) or (self.model is None):
            return None
        return AposModel(self.config, self.model)


    # Given a categorical input array, construct a dictionary for
    #  mapping the unique values in the columns of the array to integers.
    def _i_map(self, xi):
        if (len(xi.dtype) > 0):
            xi_map = [np.unique(xi[n]) for n in xi.dtype.names]
        else:
            xi_map = [np.unique(xi[:,i]) for i in range(xi.shape[1])]
        xi_sizes = [len(u) for u in xi_map]
        xi_starts = (np.cumsum(xi_sizes) - xi_sizes[0] + 1).tolist()
        return xi_map, xi_sizes, xi_starts


    # Given a categorical input array (either 2D or struct), map this
    #  array to an integer encoding matrix with the same number of
    #  columns, but unique integers assigned to each unique value.
    def _i_encode(self, xi, xi_map, xi_sizes, xi_starts):
        xi_rows = xi.shape[0]
        xi_cols = len(xi.dtype) or xi.shape[1]
        _xi = np.zeros((xi_rows, xi_cols), dtype="int32", order="C")
        for i in range(xi_cols):
            start_index = xi_starts[i]
            num_unique = xi_sizes[i]
            unique_vals = xi_map[i]
            vals = (xi[:,i:i+1] if len(xi.dtype) == 0 else xi[xi.dtype.names[i]])
            eq_val = vals == unique_vals
            # Add a column to the front that is the default if none match.
            eq_val = np.concatenate((
                np.logical_not(eq_val.max(axis=1)).reshape(xi_rows,1),
                eq_val), axis=1)
            val_indices = np.ones((xi_rows,num_unique+1), dtype="int32") * np.arange(num_unique+1)
            val_indices[:,1:] += start_index-1
            _xi[:,i] = val_indices[eq_val]
        return _xi


    # Convert all inputs to the APOS model into the expected numpy format.
    def _to_array(self, y, yi, x, xi, ax, axi, sizes):
        # Get the number of inputs.
        if   (y  is not None): nm = len(y)
        elif (yi is not None): nm = len(yi)
        elif (x  is not None): nm = len(x)
        elif (xi is not None): nm = len(xi)
        elif (sizes is not None): nm = len(sizes)
        # Make sure that all inputs are numpy arrays.
        if (y is not None):  y = np.asarray(y, dtype="float32", order="C")
        else:                y = np.zeros((nm,0), dtype="float32", order="C") 
        if (yi is not None): yi = np.asarray(yi)
        else:                yi = np.zeros((nm,0), dtype="int32", order="C")
        if (x is not None): x = np.asarray(x, dtype="float32", order="C")
        else:               x = np.zeros((nm,0), dtype="float32", order="C")
        if (xi is not None): xi = np.asarray(xi)
        else:                xi = np.zeros((nm,0), dtype="int32", order="C")
        if (sizes is not None): sizes = np.asarray(sizes, dtype="int32")
        else:                   sizes = np.zeros(0, dtype="int32")
        na = sizes.sum()
        if (ax is not None): ax = np.asarray(ax, dtype="float32", order="C")
        else:                ax = np.zeros((na,0), dtype="float32", order="C")
        if (axi is not None): axi = np.asarray(axi)
        else:                 axi = np.zeros((na,0), dtype="int32", order="C")
        # Make sure that all inputs have the expected shape.
        assert (len(y.shape) in {1,2}), f"Bad y shape {y.shape}, should be 1D or 2D matrix."
        assert (len(yi.shape) in {1,2}), f"Bad yi shape {yi.shape}, should be 1D or 2D matrix."
        assert (len(x.shape) in {1,2}), f"Bad x shape {x.shape}, should be 1D or 2D matrix."
        assert (len(xi.shape) in {1,2}), f"Bad xi shape {xi.shape}, should be 1D or 2D matrix."
        assert (len(ax.shape) in {1,2}), f"Bad ax shape {ax.shape}, should be 1D or 2D matrix."
        assert (len(axi.shape) in {1,2}), f"Bad axi shape {axi.shape}, should be 1D or 2D matrix."
        assert (len(sizes.shape) == 1), f"Bad sizes shape {sizes.shape}, should be 1D int vectora."
        # Reshape inputs to all be two dimensional (except sizes).
        if (len(y.shape) == 1): y = y.reshape((-1,1))
        if (len(yi.shape) == 1) and (len(yi.dtype) == 0): yi = yi.reshape((-1,1))
        if (len(x.shape) == 1): x = x.reshape((-1,1))
        if (len(xi.shape) == 1) and (len(xi.dtype) == 0): xi = xi.reshape((-1,1))
        if (len(ax.shape) == 1): ax = ax.reshape((-1,1))
        if ((len(axi.shape) == 1) and (len(axi.dtype) == 0)): axi = axi.reshape((-1,1))
        mdo = y.shape[1]
        mdn = x.shape[1]
        adn = ax.shape[1]
        # Handle mapping "xi" into integer encodings.
        xi_cols = len(xi.dtype) or xi.shape[1]
        if (xi_cols > 0):
            if (len(self.xi_map) == 0):
                self.xi_map, self.xi_sizes, self.xi_starts = self._i_map(xi)
            else:
                assert (xi_cols == len(self.xi_map)), f"Bad number of columns in 'xi', {xi_cols}, expected {len(self.xi_map)} columns."
            xi = self._i_encode(xi, self.xi_map, self.xi_sizes, self.xi_starts)
            mne = sum(self.xi_sizes)
        else: mne = 0
        # Handle mapping "axi" into integer encodings.
        axi_cols = len(axi.dtype) or axi.shape[1]
        if (axi_cols > 0):
            if (len(self.axi_map) == 0):
                self.axi_map, self.axi_sizes, self.axi_starts = self._i_map(axi)
            else:
                assert (axi_cols == len(self.axi_map)), f"Bad number of columns in 'axi', {axi_cols}, expected {len(self.axi_map)} columns."
            axi = self._i_encode(axi, self.axi_map, self.axi_sizes, self.axi_starts)
            ane = sum(self.axi_sizes)
        else: ane = 0
        # Handle mapping "yi" into integer encodings.
        yi_cols = len(yi.dtype) or yi.shape[1]
        if (yi_cols > 0):
            if (len(self.yi_map) == 0):
                self.yi_map, self.yi_sizes, self.yi_starts = self._i_map(yi)
            else:
                assert (yi_cols == len(self.yi_map)), f"Bad number of columns in 'yi', {yi_cols}, expected {len(self.yi_map)} columns."
            yi = self._i_encode(yi, self.yi_map, self.yi_sizes, self.yi_starts)
            yne = sum(self.yi_sizes)
        else: yne = 0
        # Handle mapping integer encoded "yi" into a single real valued y.
        if (yne > 0):
            embedded = np.concatenate((
                np.zeros((1,yne), dtype="float32"),
                np.identity(yne, dtype="float32")), axis=0)
            _y = np.zeros((nm, mdo+yne), dtype="float32")
            _y[:,:mdo] = y[:,:]
            for i in range(yi.shape[1]):
                _y[:,mdo:] += embedded[yi[:,i]]
            y = _y
            mdo += yne
        # Return all the shapes and numpy formatted inputs.
        return nm, na, mdn, mne, mdo, adn, ane, yne, y, x, xi, ax, axi, sizes


    # Fit this model.
    def fit(self, x=None, y=None, yi=None, xi=None, ax=None, axi=None,
            sizes=None, new_model=False, **kwargs):
        # Ensure that 'y' values were provided.
        assert ((y is not None) or (yi is not None)), "APOS.fit requires 'y' or 'yi' values, but neitherwere provided (use keyword argument 'y=<values>' or 'yi=<values>')."
        # Make sure that 'sizes' were provided for apositional (aggregate) inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "APOS.fit requires 'sizes' to be provided for apositional input sets (ax and axi)."
        # Get all inputs as arrays.
        nm, na, mdn, mne, mdo, adn, ane, yne, y, x, xi, ax, axi, sizes = (
            self._to_array(y, yi, x, xi, ax, axi, sizes)
        )
        # Configure this model if requested (or not already done).
        if (new_model or (self.config is None)):
            # Ensure that the config is compatible with the data.
            kwargs.update({
                "adn":adn,
                "ane":max(ane, kwargs.get("ane",0)),
                "mdn":mdn,
                "mne":max(mne, kwargs.get("mne",0)),
                "mdo":mdo,
            })
            if (max(kwargs["mdn"], kwargs["mne"]) == 0):
                kwargs["ado"] = kwargs["mdo"]
                kwargs["mdo"] = 0
                kwargs["mns"] = 0
            self._init_model(**kwargs)
        # If there are integer embeddings, expand "x" and "ax" to have space to hold those embeddings.
        if (self.config.ade > 0):
            _ax = np.zeros((ax.shape[0],ax.shape[1]+self.config.ade), dtype="float32", order="C")
            _ax[:,:ax.shape[1]] = ax
            ax, _ax = _ax, ax
        if (self.config.mde > 0) or (self.config.ado > 0):
            _x = np.zeros((x.shape[0],self.config.mdi), dtype="float32", order="C")
            _x[:,:x.shape[1]] = x
            x, _x = _x, x
        # ------------------------------------------------------------
        # If a random seed is provided, then only 2 threads can be used
        #  because nondeterministic behavior comes from reordered addition.
        if (self.seed is not None):
            if (self.config.num_threads > 2):
                import warnings
                warnings.warn("Seeding an APOS model will deterministically initialize weights, but num_threads > 2 will result in a nondeterministic model fit.")
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # ------------------------------------------------------------
        # Set up new work space for this minimization process.
        self.APOS.new_fit_config(nm, na, self.config)
        rwork = np.zeros(self.config.rwork_size, dtype="float32")
        iwork = np.zeros(self.config.iwork_size, dtype="int32")
        # Minimize the mean squared error.
        self.record = np.zeros((steps,6), dtype="float32", order="C")
        result = self.APOS.minimize_mse(self.config, self.model, rwork, iwork,
                                        ax.T, axi.T, sizes, x.T, xi.T, y.T, 
                                        steps=steps, record=self.record.T)
        assert (result[-1] == 0), f"APOS.minimize_mse returned nonzero exit code {result[-1]}."
        # Copy the updated values back into the input arrays (for transparency).
        if (self.config.mde > 0):
            _x[:,:] = x[:,:_x.shape[1]]
        if (self.config.ade > 0):
            _ax[:,:] = ax[:,:_ax.shape[1]]


    # Calling this model is an alias for 'APOS.predict'.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    # Make predictions for new data.
    def predict(self, x=None, xi=None, ax=None, axi=None, sizes=None,
                embedding=False, save_states=False, **kwargs):
        # Evaluate the model at all data.
        assert ((x is not None) or (xi is not None) or (sizes is not None)), "APOS.predict requires at least one of 'x', 'xi', or 'sizes' to not be None."
        # Make sure that 'sizes' were provided for apositional (aggregate) inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "APOS.predict requires 'sizes' to be provided for apositional input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        nm, na, mdn, mne, mdo, adn, ane, yne, _, x, xi, ax, axi, sizes = (
            self._to_array(None, None, x, xi, ax, axi, sizes)
        )
        # Embed the inputs into the purely positional form.
        ade = self.config.ade
        ads = self.config.ads
        ado = self.config.ado
        mde = self.config.mde
        mds = self.config.mds
        if (self.config.mdo != 0):
            mdo = self.config.mdo
        else:
            mdo = self.config.ado
        # Compute the true real-vector input dimensions given embeddings.
        adn += ade
        mdn += mde + ado
        # ------------------------------------------------------------
        # Initialize storage for all arrays needed at evaluation time.
        #   If there are integer embeddings, expand "ax" and "x" to have
        #   space to hold those embeddings.
        if (self.config.ade > 0):
            _ax = np.zeros((ax.shape[0],ax.shape[1]+self.config.ade), dtype="float32", order="C")
            _ax[:,:ax.shape[1]] = ax
            ax = _ax
        ay = np.zeros((na, ado), dtype="float32", order="F")
        if (self.config.mde > 0) or (self.config.ado > 0):
            _x = np.zeros((x.shape[0],self.config.mdi), dtype="float32", order="C")
            _x[:,:x.shape[1]] = x
            x = _x
        y = np.zeros((nm, mdo), dtype="float32", order="C")
        if (save_states):
            m_states = np.zeros((nm, mds, self.config.mns), dtype="float32", order="F")
            a_states = np.zeros((na, ads, self.config.ans), dtype="float32", order="F")
        else:
            m_states = np.zeros((nm, mds, 2), dtype="float32", order="F")
            a_states = np.zeros((na, ads, 2), dtype="float32", order="F")
        # ------------------------------------------------------------
        # Call the unerlying library.
        info = self.APOS.check_shape(self.config, self.model, ax.T, axi.T, sizes, x.T, xi.T, y.T)
        assert (info == 0), f"APOS.predict encountered nonzero exit code {info} when calling APOS.check_shape."
        self.APOS.embed(self.config, self.model, axi.T, xi.T, ax.T, x.T)
        result = self.APOS.evaluate(self.config, self.model, ax.T, ay, sizes,
                                    x.T, y.T, a_states, m_states, info)
        assert (result[-1] == 0), f"APOS.evaluate returned nonzero exit code {result[-1]}."
        # Save the states if that's 
        if (save_states):
            self.m_states = m_states
            self.a_states = a_states
        # If there are embedded y values in the output, return them to the format at training time.
        if (len(self.yi_map) > 0) and (not embedding):
            yne = sum(self.yi_sizes)
            _y = [y[:,i] for i in range(y.shape[1]-yne)]
            for i in range(len(self.yi_map)):
                start = self.yi_starts[i]
                size = self.yi_sizes[i]
                _y.append(
                    self.yi_map[i][np.argmax(y[:,start:start+size], axis=1)]
                )
            return np.asarray(_y).T
        elif (embedding and (len(self.yi_map) == 0)):
            return m_states[:,:,-2]
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
                "xi_map"    : [l.tolist() for l in self.xi_map],
                "xi_sizes"  : self.xi_sizes,
                "xi_starts" : self.xi_starts,
                "axi_map"    : [l.tolist() for l in self.axi_map],
                "axi_sizes"  : self.axi_sizes,
                "axi_starts" : self.axi_starts,
                "yi_map"    : [l.tolist() for l in self.yi_map],
                "yi_sizes"  : self.yi_sizes,
                "yi_starts" : self.yi_starts,
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
            elif (key[:2] in {"xi_","axi_","yi_"}):
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

    # ----------------------------------------------------------------
    #  Enable debugging option "-fcheck=bounds".
    import fmodpy
    fmodpy.config.f_compiler_args = "-fPIC -shared -O3 -fcheck=bounds"
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
    state_dim = 16
    num_states = 4
    steps = 1000
    num_threads = None
    np.random.seed(seed)

    TEST_FIT_SIZE = False
    TEST_SAVE_LOAD = True
    TEST_INT_INPUT = False
    TEST_APOSITIONAL = False
    TEST_VARIED_SIZE = False
    SHOW_VISUALS = True


    if TEST_FIT_SIZE:
        # Apositional model settings.
        dim_a_numeric = 4
        dim_a_embedding = 32
        num_a_embeddings = 32
        dim_a_state = 128
        num_a_states = 1
        dim_a_out = 64
        # Model settings.
        dim_m_numeric = 8
        dim_m_embedding = 0
        num_m_embeddings = 0
        dim_m_state = 64
        num_m_states = 8
        dim_m_out = 1
        # Number of points going into each model.
        na = 10000000
        nm = 100000
        num_threads = 100
        # Initialize the model and its fit configuration.
        m = APOS(
            adn=dim_a_numeric, ade=dim_a_embedding, ane=num_a_embeddings,
            ads=dim_a_state, ans=num_a_states, ado=dim_a_out,
            mdn=dim_m_numeric, mde=dim_m_embedding, mne=num_m_embeddings,
            mds=dim_m_state, mns=num_m_states, mdo=dim_m_out,
            num_threads=num_threads, seed=seed,
        )
        m.APOS.new_fit_config(nm, na, m.config)
        print()
        print(m)


    if (not SHOW_VISUALS):
        class Plot:
            def __init__(self, *args, **kwargs): pass
            def __getattr__(self, *args, **kwargs):
                return lambda *args, **kwargs: None

    if TEST_SAVE_LOAD:
        # Try saving an untrained model.
        m = APOS()
        print("Empty model:")
        print("  str(model) =", str(m))
        print()
        m.save("testing_empty_save.json")
        m.load("testing_empty_save.json")
        from util.approximate import PLRM
        m = APOS(mdn=2, mds=state_dim, mns=num_states, mdo=1, seed=seed,
                 num_threads=num_threads, steps=steps, 
                 ) # discontinuity=-1000.0) # initial_step=0.01)
        print("Initialized model:")
        print(m)
        print()
        # Create the test plot.
        x = np.asarray(well_spaced_box(n, 2), dtype="float32", order="C")
        # x[:,0] /= 2
        y = f(x).astype("float32")
        # Fit the model.
        m.fit(x.copy(), y.copy())
        # Add the data and the surface of the model to the plot.
        p = Plot()
        x_min_max = np.asarray([x.min(axis=0), x.max(axis=0)]).T
        p.add("Data", *x.T, y)
        # p.add("Normalized data", *x_fit.T, y_fit)
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
        p.add("Loaded values", *x.T, m(x)[:,0]+0.05, color=1, marker_size=4)
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
        m = APOS(mdn=2, mds=state_dim, mns=num_states, mdo=1, mde=3, mne=2, seed=seed, steps=steps, num_threads=num_threads)
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


    if TEST_APOSITIONAL:
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
        m = APOS(mdn=0, adn=ax.shape[1], ado=2, mdo=all_y.shape[1], 
                 ads=state_dim, ans=num_states, mds=state_dim, mns=num_states,
                 ane=len(np.unique(axi.flatten())), mne=len(np.unique(all_xi.flatten())),
                 num_threads=num_threads, seed=seed)
        m.fit(ax=ax.copy(), axi=axi, sizes=sizes, xi=all_xi, y=all_y.copy(), 
              steps=steps, num_threads=num_threads, seed=seed)
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


    if TEST_VARIED_SIZE:
        print("Creating data..")
        for test in range(100):
            print("sizes test: ", test, end="\r")
            sizes = np.random.randint(5,20,size=(10))
            na = sizes.sum()
            nm = sizes.size
            ax = np.random.random(size=(na,2))
            x = well_spaced_box(nm, 2)
            y = f(x)
            start = 0
            for i in range(len(sizes)):
                end = start + sizes[i]
                y[i] += ax[start:end].max()
                start = end
            # Fit a model.
            m = APOS(seed=seed, num_threads=num_threads, steps=1)
            m.fit(x=x.copy(), y=y.copy(), ax=ax.copy(), sizes=sizes)


# 2022-04-03 15:03:09
# 
        #########################################################################
        # # Generate visual of the embeddings.                                  #
        # _ = m.unpack()                                                        #
        # embeddings = _.m_embeddings.T                                         #
        # output_vecs = _.m_output_vecs.T                                       #
        # print()                                                               #
        # print("Embeddings:")                                                  #
        # print(embeddings)                                                     #
        # print()                                                               #
        # print("Output vecs:")                                                 #
        # print(output_vecs)                                                    #
        # print()                                                               #
        #                                                                       #
        # p = Plot("Embeddings")                                                #
        # # Reduce the dimension to 3 if it is higher (using singular vectors). #
        # if (embeddings.shape[1] > 3):                                         #
        #     from tlux.math import svd                                         #
        #     _, projection = svd(embeddings.copy())                            #
        #     embeddings = np.matmul(embeddings, projection[:3].T)              #
        # # Raise the dimension from 1 to 2.                                    #
        # elif (embeddings.shape[1] == 1):                                      #
        #     embeddings = np.asarray([[v,0] for v in embeddings.flatten()])    #
        # # Add the embedding vectors to the plot.                              #
        # for vec in embeddings:                                                #
        #     p.add(str(vec), *[[0,v] for v in vec], mode="lines")              #
        # p.show(append=True, show=False)                                       #
        #########################################################################


# 2022-04-03 19:43:37
# 
        ###########################################################################################
        # if False:                                                                               #
        #     m = APOS(mdn=0, adn=ax.shape[1], ado=2, mdo=0, #all_y.shape[1],                     #
        #              ads=state_dim, ans=num_states, # mds=state_dim, mns=num_states,            #
        #              ane=len(np.unique(axi.flatten())), # mne=len(np.unique(all_xi.flatten())), #
        #              num_threads=num_threads, seed=seed)                                        #
        #     fit_ax = np.array(ax, dtype="float32", order="C")                                   #
        #     fit_y = np.array(all_x, dtype="float32", order="C")                                 #
        #     print("fit_y.mean(axis=0): ", fit_y.mean(axis=0))                                   #
        #     print()                                                                             #
        #     print(m.config_str())                                                               #
        #     print()                                                                             #
        #     m.fit(ax=fit_ax, axi=axi, sizes=sizes, y=fit_y,                                     #
        #           steps=1000, num_threads=num_threads, seed=seed)                               #
        #     fx = m(ax=ax.copy(), axi=axi, sizes=sizes)                                          #
        #     print("fit_y.mean(axis=0): ", fit_y.mean(axis=0))                                   #
        #     print("m.unpack().ay_shift: ", m.unpack().ay_shift)                                 #
        #     print(" m.unpack().y_shift: ", m.unpack().y_shift)                                  #
        #     print()                                                                             #
        #     print("all_x: ")                                                                    #
        #     print(all_x)                                                                        #
        #     print()                                                                             #
        #     print("fx: ")                                                                       #
        #     print(fx)                                                                           #
        #     print()                                                                             #
        #     print("all_x - fx: ")                                                               #
        #     print(all_x - fx)                                                                   #
        # else:                                                                                   #
        ###########################################################################################
