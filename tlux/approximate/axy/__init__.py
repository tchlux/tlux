import os, re
import numpy as np

# Build a class that contains pointers to the model internals, allowing
#  python attribute access to all of the different components of the models.
class AxyModel:
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
            f"AXY model ({self.config.total_size} parameters) [{byte_size}]\n"+
            (" aggregator model\n"+
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
            (" model\n"+
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


# Class for calling the underlying AXY model code.
class AXY:
    # Make the string function return the unpacked model.
    def __str__(self): return str(self.unpack())

    # Initialize a new AXY model.
    def __init__(self, **kwargs):
        try:
            # Store the Fortran module as an attribute.
            from tlux.approximate.axy.axy import axy
            self.AXY = axy
        except:
            from tlux.setup import build_axy
            try:
                self.AXY = build_axy().axy
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
        self.embedding_transform = np.zeros(0, dtype="float32")
        # Default descriptors for categorical inputs.
        self.axi_map = []
        self.xi_map = []
        self.yi_map = []
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
            self.AXY.init_model(self.config, self.model, seed=self.seed)


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
        return AxyModel(self.config, self.model)


    # Given an exit code, check to make sure it is 0 (good), if not then read
    #  the source file to return a reason for the code if it can be found.
    def _check_code(self, exit_code, method):
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
            # Raise an assertion error.
            assert (exit_code == 0), f"AXY.{method} returned nonzero exit code {exit_code}. {reason}"


    # Given a categorical input array, construct a dictionary for
    #  mapping the unique values in the columns of the array to integers.
    def _i_map(self, xi):
        if (len(xi.dtype) > 0):
            xi_map = [np.unique(xi[n]) for n in xi.dtype.names]
        else:
            xi_map = [np.unique(xi[:,i]) for i in range(xi.shape[1])]
        return xi_map


    # Given a categorical input array (either 2D or struct), map this
    #  array to an integer encoding matrix with the same number of
    #  columns, but unique integers assigned to each unique value.
    # 
    # TODO: This is very slow and doesn't use any parallelism right now,
    #       could be much faster for large data by paralellizing.
    def _i_encode(self, xi, xi_map):
        xi_rows = xi.shape[0]
        xi_cols = len(xi.dtype) or xi.shape[1]
        _xi = np.zeros((xi_rows, xi_cols), dtype="int32", order="C")
        base = 1
        for i in range(xi_cols):
            # Construct the integer map, retrieve the column of unique values.
            # The integer map should not have any numbers shared across yi columns!
            int_map = {xi_map[i][j]:base+j for j in range(len(xi_map[i]))}
            base += len(xi_map[i])
            vals = (xi[:,i:i+1] if len(xi.dtype) == 0 else xi[xi.dtype.names[i]])
            # Assign all the integers.
            for j in range(xi_rows):
                _xi[j,i] = int_map.get(vals[j,0], 0)
        return _xi


    # Convert all inputs to the AXY model into the expected numpy format.
    def _to_array(self, ax, axi, sizes, x, xi, y, yi):
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
        assert (len(sizes.shape) == 1), f"Bad sizes shape {sizes.shape}, should be 1D int vector."
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
                self.xi_map = self._i_map(xi)
            else:
                assert (xi_cols == len(self.xi_map)), f"Bad number of columns in 'xi', {xi_cols}, expected {len(self.xi_map)} columns."
            xi = self._i_encode(xi, self.xi_map)
            mne = sum(map(len, self.xi_map))
        else: mne = 0
        # Handle mapping "axi" into integer encodings.
        axi_cols = len(axi.dtype) or axi.shape[1]
        if (axi_cols > 0):
            if (len(self.axi_map) == 0):
                self.axi_map = self._i_map(axi)
            else:
                assert (axi_cols == len(self.axi_map)), f"Bad number of columns in 'axi', {axi_cols}, expected {len(self.axi_map)} columns."
            axi = self._i_encode(axi, self.axi_map)
            ane = sum(map(len, self.axi_map))
        else: ane = 0
        # Handle mapping "yi" into integer encodings.
        yi_cols = len(yi.dtype) or yi.shape[1]
        if (yi_cols > 0):
            if (len(self.yi_map) == 0):
                self.yi_map = self._i_map(yi)
            else:
                assert (yi_cols == len(self.yi_map)), f"Bad number of columns in 'yi', {yi_cols}, expected {len(self.yi_map)} columns."
            yi = self._i_encode(yi, self.yi_map)
            yne = sum(map(len, self.yi_map))
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
    # TODO: When sizes for Aggregator are set, but aggregate data has
    #       zero shape, then reset the aggregator sizes to be zeros.
    def fit(self, ax=None, axi=None, sizes=None, x=None, xi=None,
            y=None, yi=None, yw=None, new_model=False, **kwargs):
        # Ensure that 'y' values were provided.
        assert ((y is not None) or (yi is not None)), "AXY.fit requires 'y' or 'yi' values, but neitherwere provided (use keyword argument 'y=<values>' or 'yi=<values>')."
        # Make sure that 'sizes' were provided for aggregate inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.fit requires 'sizes' to be provided for aggregated input sets (ax and axi)."
        # Get all inputs as arrays.
        nm, na, mdn, mne, mdo, adn, ane, yne, y, x, xi, ax, axi, sizes = (
            self._to_array(ax, axi, sizes, x, xi, y, yi)
        )
        # TODO: Move yw into the _to_array function.
        # Convert yw to a numpy array (if it is not already).
        if (yw is None):
            yw = np.zeros((nm,0), dtype="float32", order="C")
        else:
            yw = np.asarray(np.asarray(yw, dtype="float32").reshape((nm,-1)), order="C")
        assert (yw.shape[1] in {0, 1, mdo}), f"Weights for points 'yw' {yw.shape} must have 1 column{' or '+str(mdo)+' columns' if (mdo > 0) else ''}."
        # Configure this model if requested (or not already done).
        self._init_kwargs.update(kwargs)
        kwargs = self._init_kwargs
        if (new_model or (self.config is None)):
            # Ensure that the config is compatible with the data.
            kwargs.update({
                "adn":adn,
                "ane":max(ane, kwargs.get("ane",0)),
                "ado":kwargs.get("ado", (mdo if max(adn,ane) > 0 else 0)),
                "mdn":mdn,
                "mne":max(mne, kwargs.get("mne",0)),
                "mdo":mdo,
            })
            self._init_model(**kwargs)
        else:
            # Set any configuration keyword arguments.
            for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
                if (kwargs[n] is not None):
                    setattr(self.config, n, kwargs[n])            
        # TODO: Should the following two conditions be in the _to_array function as well?
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
                warnings.warn("Seeding an AXY model will deterministically initialize weights, but num_threads > 2 will result in a nondeterministic model fit.")
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # ------------------------------------------------------------
        # Set up new work space for this minimization process.
        self.AXY.new_fit_config(nm, na, self.config)
        rwork = np.zeros(self.config.rwork_size, dtype="float32")
        iwork = np.zeros(self.config.iwork_size, dtype="int32")
        # Minimize the mean squared error.
        self.record = np.zeros((steps,6), dtype="float32", order="C")
        result = self.AXY.minimize_mse(self.config, self.model, rwork, iwork,
                                        ax.T, axi.T, sizes, x.T, xi.T, y.T, yw.T,
                                        steps=steps, record=self.record.T)
        # Check for a nonzero exit code.
        self._check_code(result[-1], "minimize_mse")
        # Copy the updated values back into the input arrays (for transparency).
        if (self.config.mde > 0):
            _x[:,:] = x[:,:_x.shape[1]]
        if (self.config.ade > 0):
            _ax[:,:] = ax[:,:_ax.shape[1]]
        # Store the multiplier to be used in embeddings (to level the norm contribution).
        if (self.config.mdo > 0):
            last_weights = self.model[self.config.msov-1:self.config.meov].reshape(self.config.mdso, self.config.mdo, order="F")
        else:
            last_weights = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.adso, self.config.ado, order="F")
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
                embedding=False, save_states=False, raw_scores=False, **kwargs):
        # Evaluate the model at all data.
        assert ((x is not None) or (xi is not None) or (sizes is not None)), "AXY.predict requires at least one of 'x', 'xi', or 'sizes' to not be None."
        assert (not (embedding and raw_scores)), "AXY.predict cannot provide both 'embedding=True' *and* 'raw_scores=True'."
        # Make sure that 'sizes' were provided for aggregate inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.predict requires 'sizes' to be provided for aggregated input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        nm, na, mdn, mne, mdo, adn, ane, yne, _, x, xi, ax, axi, sizes = (
            self._to_array(ax, axi, sizes, x, xi, None, None)
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
            _x = np.zeros((x.shape[0],x.shape[1]+self.config.mde+self.config.ado),
                          dtype="float32", order="C")
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
        info = self.AXY.check_shape(self.config, self.model, ax.T, axi.T, sizes, x.T, xi.T, y.T)
        # Check for a nonzero exit code.
        self._check_code(info, "check_shape")
        # Embed the categorical inputs as numeical inputs.
        self.AXY.embed(self.config, self.model, axi.T, xi.T, ax.T, x.T)
        # Evaluate the model.
        result = self.AXY.evaluate(self.config, self.model, ax.T, ay, sizes,
                                    x.T, y.T, a_states, m_states, info)
        self._check_code(result[-1], "evaluate")
        # Save the states if that's requested.
        if (save_states):
            self.m_states = m_states
            self.a_states = a_states
        # If embeddings are desired, multiply the last state by the 2-norm
        #  of output weights for each component of that last embedding.
        if (embedding):
            # Store the "last_state" representation of data before output,
            #  as well as the "last_weights" that preceed output.
            if (self.config.mdo > 0):
                if (self.config.mns > 0):
                    last_state = m_states[:,:,-2]
                else:
                    last_state = x
            else:
                if (self.config.ans > 0):
                    last_state = a_states[:,:,-2]
                else:
                    last_state = ax
            # Rescale the last state to linearly level according to outputs.
            if (self.embedding_transform.size > 0):
                last_state = last_state * self.embedding_transform
            # Return the last state.
            return last_state
        # If there are categorical outputs, then select by taking the max magnitude output.
        # TODO: The max might not have the same meaning for each category, instead
        #       do a sliding window k-nearest-neighbor probability assignment for outcomes.
        elif ((len(self.yi_map) > 0) and (not raw_scores)):
            yne = sum(map(len, self.yi_map))
            ynn = y.shape[1]-yne
            _y = [y[:,i] for i in range(ynn)]
            past_indices = ynn
            for i in range(len(self.yi_map)):
                start = past_indices
                size = len(self.yi_map[i])
                _y.append(
                    self.yi_map[i][np.argmax(y[:,start:start+size], axis=1)]
                )
                past_indices += size
            return np.asarray(_y, dtype=object).T
        # Otherwise simply return the numeric predicted outputs by the model.
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
                "embedding_transform" : self.embedding_transform.tolist(),
                "xi_map"    : [l.tolist() for l in self.xi_map],
                "axi_map"    : [l.tolist() for l in self.axi_map],
                "yi_map"    : [l.tolist() for l in self.yi_map],
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
            elif (type(value) is list): 
                value = np.asarray(value, dtype="float32")
            setattr(self, key, value)
        # Convert the the dictionary configuration into the correct type.
        if (type(self.config) is dict):
            self.config = self.AXY.MODEL_CONFIG(**self.config)
        # Return self in case an assignment was made.
        return self


if __name__ == "__main__":
    print("_"*70)
    print(" TESTING AXY MODULE")

    # ----------------------------------------------------------------
    #  Enable debugging option "-fcheck=bounds".
    # import fmodpy
    # fmodpy.config.f_compiler_args = "-fPIC -shared -O3 -fcheck=bounds"
    # fmodpy.config.link_blas = "-framework Accelerate"
    # fmodpy.config.link_lapack = "-framework Accelerate"
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

    # Genreate source data.
    base_x = well_spaced_box(n, 2)
    x_min_max = np.vstack((np.min(base_x,axis=0), np.max(base_x, axis=0))).T
    base_y = f(base_x)

    # Create all data.
    x = np.concatenate((base_x, base_x), axis=0)
    xi = np.concatenate((np.ones(len(base_x)),2*np.ones(len(base_x)))).reshape((-1,1)).astype("int32")
    ax = x.reshape((-1,1)).copy()
    axi = (np.ones(x.shape, dtype="int32") * (np.arange(x.shape[1])+1)).reshape(-1,1)
    sizes = np.ones(x.shape[0], dtype="int32") * 2
    y = np.concatenate((base_y, np.cos(np.linalg.norm(base_x,axis=1))), axis=0)
    y = y.reshape((y.shape[0],-1))
    yi = np.asarray([
        np.where(
            y[:,0] <= np.percentile(y[:,0], 50),
            'b',
            't'
        ),
        np.where(
            y[:,0] <= np.percentile(y[:,0], 20),
            's',
            np.where(
                y[:,0] <= np.percentile(y[:,0], 80),
                'm',
                'l'
            )
        ),
    ]).T

    # Train a new model if there is not one saved.
    if (not os.path.exists('temp-model.json')):
        # Initialize a new model.
        print("Fitting model..")
        m = AXY(
            ads=state_dim,
            ans=num_states // 2,
            mds=state_dim,
            mns=num_states // 2,
            num_threads=num_threads, seed=seed,
            early_stop = False,
            # basis_replacement = True,
            # orthogonalizing_step_frequency = 200,
            # keep_best = False,
            # step_replacement = 0.00,
            # rescale_y = False,
        )
        m.fit(
            ax=ax.copy(),
            axi=axi,
            sizes=sizes,
            x=x.copy(),
            xi=xi,
            y=y.copy(),
            yi=yi,
            steps=steps,
        )
        # Save and load the model.
        m.save("temp-model.json")

    # Load the saved model.
    m = AXY()
    m.load("temp-model.json")
    os.remove("temp-model.json")

    # Print the model.
    print()
    print(m)

    # Evaluate the model and compare with the data provided for training.
    # TODO: Add some evaluations of the categorical outputs.
    # fy = m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)
    yy = np.concatenate((y.astype(object), yi.astype(object)), axis=1)

    # Generate the plot of the results.
    print("Adding to plot..")
    p = Plot()
    p.add("xi=1 true", *x.T, y[:len(y)//2,0], color=0, group=0)
    p.add("xi=2 true", *x.T, y[len(y)//2:,0], color=1, group=1)
    
    # from tlux.approximate.balltree import BallTree
    # tree = BallTree(m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, embedding=True), build=False)
    def fhat(x, i=1):
        xi = i * np.ones((len(x),1),dtype="int32")
        ax = x.reshape((-1,1)).copy()
        axi = (np.ones(x.shape, dtype="int32") * (np.arange(x.shape[1])+1)).reshape(-1,1)
        sizes = np.ones(x.shape[0], dtype="int32") * 2
        return m(x=x, xi=xi, ax=ax, axi=axi, sizes=sizes)[:,0]
        # # Use the tree to lookup the nearest neighbor.
        # emb = m(x=x, xi=xi, ax=ax, axi=axi, sizes=sizes, embedding=True)
        # i = tree(emb, return_distance=False)
        # return y[i,0]
    p.add_func("xi=1", lambda x: fhat(x, 1), [-0.1,1.1], [-0.1,1.1], vectorized=True, color=3, opacity=0.8, group=0, plot_points=3000) #, mode="markers", shade=True)
    p.add_func("xi=2", lambda x: fhat(x, 2), [-0.1,1.1], [-0.1,1.1], vectorized=True, color=2, opacity=0.8, group=1, plot_points=3000) #, mode="markers", shade=True)

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
