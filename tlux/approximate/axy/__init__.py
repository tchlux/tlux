import os, re, math
import numpy as np


# Build a class that contains pointers to the model internals, allowing
#  python attribute access to all of the different components of the models.
class AxyModel:
    def __init__(self, config, model, clock_rate=2000000000):
        self.config = config
        self.model = model
        self.clock_rate = clock_rate
        self.a_embeddings  = self.model[self.config.asev-1:self.config.aeev].reshape(self.config.ade, self.config.ane, order="F")
        self.a_input_vecs  = self.model[self.config.asiv-1:self.config.aeiv].reshape(self.config.adi, self.config.ads, order="F")
        self.a_input_shift = self.model[self.config.asis-1:self.config.aeis].reshape(self.config.ads, order="F")
        self.a_state_vecs  = self.model[self.config.assv-1:self.config.aesv].reshape(self.config.ads, self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_state_shift = self.model[self.config.asss-1:self.config.aess].reshape(self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_output_vecs = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.adso, self.config.ado+1, order="F")
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
        if hasattr(self.config, attr):
            return getattr(self.config, attr)
        else:
            return self.__getattribute__(attr)

    # Allow the "." operator to access attributes of this model and its configuration.
    def __getattr__(self, attr):
        if hasattr(self.config, attr):
            return getattr(self.config, attr)
        else:
            return self.__getattribute__(attr)

    # Create a string summarizing how time has been spent for this model.
    def _timer_summary_string(self):
        config = self.config
        clock_rate = self.clock_rate
        # Show the summary of time spent on different operations.
        time_vars = [
            ("int", "initialize"),
            ("fit", "fit"),
            ("nrm", "normalize"),
            ("gen", "fetch data"),
            ("emb", "embed"),
            ("evl", "evaluate"),
            ("grd", "gradient"),
            ("rat", "rate update"),
            ("opt", "step vars"),
            ("con", "condition"),
            ("enc", "encode")
        ]
        max_p, max_n = max(time_vars, key=lambda pn: getattr(config, "w"+pn[0]))
        total = getattr(config, "w"+max_p)
        if (total <= 0): return ""
        max_len = max((len(n) for (p,n) in time_vars))
        # Generate a summary line for each item.
        time_str = f" time category    sec {max_n+'%':>12s}   speedup\n"
        for (p, n) in time_vars:
            v = getattr(config, "w"+p)
            pv = getattr(config, "c"+p)
            time_str += f"  {n:{max_len}s}   {v/(clock_rate if clock_rate > 0 else 1):.1e}s   ({100.0*v/(total if total > 0 else 1):5.1f}%)   [{pv/(v if v > 0 else 1):.1f}x]\n"
        return "\n" + time_str + "\n"

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
            f"  input dimension  {self.config.adn}{' + '+str(self.config.ade) if self.config.ade > 0 else ''}\n"+
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
            f"  input dimension  {self.config.mdn}{' + '+str(self.config.mde) if self.config.mde > 0 else ''}{' + '+str(self.config.ado) if self.config.ado > 0 else ''}\n"+
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
        ) + self._timer_summary_string()


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
            initial_shift_range = kwargs.get("initial_shift_range", None)
            initial_output_scale = kwargs.get("initial_output_scale", None)
            self.AXY.init_model(self.config, self.model, seed=self.seed,
                                initial_shift_range=initial_shift_range,
                                initial_output_scale=initial_output_scale)


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
        return AxyModel(self.config, self.model, clock_rate=self.AXY.clock_rate)


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
        sort_key = lambda i: i if isinstance(i,str) else str(i)
        # Generate the map (ordered list of unique values).
        if (len(xi.dtype) > 0):
            xi_list = [sorted(set(xi[n].tolist()), key=sort_key) for n in xi.dtype.names]
        else:
            xi_list = [sorted(set(xi[:,i].tolist()), key=sort_key) for i in range(xi.shape[1])]
        # Generate the lookup table (value -> integer index).
        base = 1
        xi_map = []
        for i, xij_list in enumerate(xi_list):
            # If a categorical input has no variance, it will have no embedding.
            if (len(xij_list) <= 1):
                xi_map.append({})
            # Otherwise, add entries for mapping the unique values to integers.
            else:
                xi_map.append(
                    {v:base+j for j,v in enumerate(xij_list)}
                )
                base += len(xij_list)
        # Return the map and the lookup.
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
        for i, i_map in enumerate(xi_map):
            # Assign all the integer embeddings.
            values = (xi[:,i].tolist() if len(xi.dtype) == 0 else xi[xi.dtype.names[i]].tolist())
            for j,v in enumerate(values):
                _xi[j,i] = i_map.get(v, 0)
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
                self.yi_inv_map = [{v:k for (k,v) in m.items()} for m in self.yi_map]
            else:
                assert (yi_cols == len(self.yi_map)), f"Bad number of columns in 'yi', {yi_cols}, expected {len(self.yi_map)} columns."
            yi = self._i_encode(yi, self.yi_map)
            yne = sum(map(len, self.yi_map))
        else: yne = 0
        # Handle mapping integer encoded "yi" into a single real valued y.
        if (yne > 0):
            # Use a regular simplex to construct equally spaced embeddings for the categories.
            from tlux.math import regular_simplex
            embedded = np.concatenate((
                np.zeros((1,yne-1), dtype="float32"),
                regular_simplex(yne).astype("float32")), axis=0)
            _y = np.zeros((nm, mdo+yne-1), dtype="float32")
            _y[:,:mdo] = y[:,:]
            for i in range(yi.shape[1]):
                _y[:,mdo:] += embedded[yi[:,i]]
            y = _y
            mdo += yne-1
            self.yi_embeddings = embedded[1:]
        # Return all the shapes and numpy formatted inputs.
        return nm, na, mdn, mne, mdo, adn, ane, yne, y, x, xi, ax, axi, sizes


    # Fit this model.
    # TODO: When sizes for Aggregator are set, but aggregate data has
    #       zero shape, then reset the aggregator sizes to be zeros.
    def fit(self, ax=None, axi=None, sizes=None, x=None, xi=None,
            y=None, yi=None, yw=None, new_model=False, nm=None, na=None, **kwargs):
        # Ensure that 'y' values were provided.
        assert ((y is not None) or (yi is not None)), "AXY.fit requires 'y' or 'yi' values, but neitherwere provided (use keyword argument 'y=<values>' or 'yi=<values>')."
        # Make sure that 'sizes' were provided for aggregate inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.fit requires 'sizes' to be provided for aggregated input sets (ax and axi)."
        # Get all inputs as arrays.
        full_nm, full_na, mdn, mne, mdo, adn, ane, yne, y, x, xi, ax, axi, sizes = (
            self._to_array(ax, axi, sizes, x, xi, y, yi)
        )
        if (na is None): na = full_na
        else:            na = min(na, full_na)
        if (nm is None): nm = full_nm
        else:            nm = min(nm, full_nm)
        # Make sure NA is only as large as it practically needs to be.
        na = min(na, sum(sizes[np.argsort(sizes)[-nm:]]))
        # TODO: Move yw into the _to_array function?
        # Convert yw to a numpy array (if it is not already).
        if (yw is None):
            yw = np.zeros((full_nm,0), dtype="float32", order="C")
        else:
            yw = np.asarray(np.asarray(yw, dtype="float32").reshape((full_nm,-1)), order="C")
        assert (yw.shape[1] in {0, 1, mdo}), f"Weights for points 'yw' {yw.shape} must have 1 column{' or '+str(mdo)+' columns' if (mdo > 0) else ''}."
        # Configure this model if requested (or not already done).
        self._init_kwargs.update(kwargs)
        kwargs = self._init_kwargs
        if (new_model or (self.config is None)):
            # Ensure that the config is compatible with the data.
            kwargs.update({
                "adn":adn,
                "ane":max(ane, kwargs.get("ane",0)),
                "ado":kwargs.get("ado", (mdo if (kwargs.get("mdo",None) == 0) else None)),
                "mdn":mdn,
                "mne":max(mne, kwargs.get("mne",0)),
                "mdo":kwargs.get("mdo", mdo),
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
                warnings.warn("Seeding an AXY model will deterministically initialize weights, but num_threads > 2 will result in a nondeterministic model fit.")
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # ------------------------------------------------------------
        # Set up new work space for this minimization process.
        self.AXY.new_fit_config(nm=nm, na=na, nmt=x.shape[0], nat=ax.shape[0],
                                adi=axi.shape[1], mdi=xi.shape[1],
                                seed=self.seed, config=self.config)
        rwork = np.ones(self.config.rwork_size, dtype="float32")  # beware of allocation, heap vs stack
        iwork = np.ones(self.config.iwork_size, dtype="int32")
        # Minimize the mean squared error.
        self.record = np.zeros((steps,6), dtype="float32", order="C")
        result = self.AXY.fit_model(self.config, self.model, rwork, iwork,
                                    ax.T, axi.T, sizes, x.T, xi.T, y.T, yw.T,
                                    steps=steps, record=self.record.T)
        # Check for a nonzero exit code.
        self._check_code(result[-1], "fit_model")
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
        # Initialize holder for y output.
        y = np.zeros((nm, mdo), dtype="float32", order="C")
        # ------------------------------------------------------------
        # Call the unerlying library to make sure input shapes are appropriate.
        info = self.AXY.check_shape(self.config, self.model, ax.T, axi.T, sizes, x.T, xi.T, y.T)
        # Check for a nonzero exit code.
        self._check_code(info, "check_shape")
        # ------------------------------------------------------------
        # Initialize storage for all arrays needed at evaluation time.
        #   If there are integer embeddings, expand "ax" and "x" to have
        #   space to hold those embeddings.
        if (self.config.ade > 0):
            _ax = np.zeros((ax.shape[0],ax.shape[1]+self.config.ade), dtype="float32", order="C")
            _ax[:,:ax.shape[1]] = ax
            ax = _ax
        else:
            ax = ax.copy()
        ay = np.zeros((na, ado+1), dtype="float32", order="F")
        if (self.config.mdo > 0) and ((self.config.mde > 0) or (self.config.ado > 0)):
            _x = np.zeros((x.shape[0],x.shape[1]+self.config.mde+self.config.ado),
                          dtype="float32", order="C")
            _x[:,:x.shape[1]] = x
            x = _x
        else:
            x = x.copy()
        # If all internal states are being saved, the make more space, otherwise only two copies are needed.
        if (save_states):
            a_states = np.zeros((na, ads, self.config.ans), dtype="float32", order="F")
            m_states = np.zeros((nm, mds, self.config.mns), dtype="float32", order="F")
        else:
            a_states = np.zeros((na, ads, 2), dtype="float32", order="F")
            m_states = np.zeros((nm, mds, 2), dtype="float32", order="F")
        # ------------------------------------------------------------
        # Embed the categorical inputs as numeical inputs.
        self.AXY.embed(self.config, self.model, axi.T, xi.T, ax.T, x.T)
        # Evaluate the model.
        result = self.AXY.evaluate(self.config, self.model, ax.T, ay, sizes,
                                   x.T, y.T, a_states, m_states, info)
        self._check_code(result[-1], "evaluate")
        # Save the states if that's requested.
        if (save_states):
            self.a_states = a_states
            self.ay = ay
            self.m_states = m_states
        else:
            self.a_states = None
            self.ay = None
            self.m_states = None
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
            else:  # There is only an aggregator, no model follows.
                if (self.config.ans > 0):
                    state = a_states[:,:,-2]
                else:
                    state = ax
                # Collapse the aggregate embeddings into their mean.
                last_state = np.zeros((sizes.shape[0], state.shape[1]), dtype="float32")
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
            yne = sum(map(len, self.yi_map))
            ynn = y.shape[1]-(yne-1)
            _y = [y[:,i] for i in range(ynn)]
            y = np.matmul(y[:,ynn:], self.yi_embeddings.T)
            past_indices = 0
            for i in range(len(self.yi_map)):
                start = past_indices
                size = len(self.yi_map[i])
                _y.append(
                    [self.yi_inv_map[i][start+j] for j in 1+np.argmax(y[:,start:start+size], axis=1)]
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
                "xi_map" : [list(m.items()) for m in self.xi_map],
                "axi_map" : [list(m.items()) for m in self.axi_map],
                "yi_map" : [list(m.items()) for m in self.yi_map],
                "yi_inv_map" : [list(m.items()) for m in getattr(self, "yi_inv_map", [])],
                "yi_embeddings" : getattr(self, "yi_embeddings", np.asarray([])).tolist(),
            }))


    # Load this model from a path (after having been saved).
    def load(self, path):
        # Read the file.
        import builtins, json
        with open(path, "r") as f:
            attrs = json.loads(f.read())
        # Load the attributes of the model.
        for key, value in attrs.items():
            # Convert the categorical map keys to the appropriate type.
            if (key.endswith("_map")):
                value = list(map(dict, value))
            # Otherwise all lists are actually numpy arrays (weights).
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

    # # Remove the compiled object if modifications have been made to sources.
    # import os
    # f9 = os.path.dirname(os.path.abspath(__file__))
    # f9 = [os.path.join(f9,p) for p in os.listdir(f9) if p.endswith(".f90")]
    # so = os.path.expanduser("~/Git/tlux/tlux/approximate/axy/axy/axy.arm64.so")
    # if os.path.exists(so) and (max(map(os.path.getmtime, f9)) > os.path.getmtime(so)):
    #     os.remove(so)

    import fmodpy
    AXY_MOD = fmodpy.fimport(
        input_fortran_file = "axy.f90",
        dependencies = ["random.f90", "matrix_operations.f90", "sort_and_select.f90", "axy.f90"],
        name = "axy",
        blas = True,
        lapack = True,
        omp = True,
        wrap = True,
        # rebuild = True,
        verbose = False,
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

    functions = [f1, f2,]# f3]

    seed = 0
    np.random.seed(seed)

    n = 2**7
    d = 2
    new_model = True
    use_a = True
    agg_dim = 64
    agg_states = 2
    use_x = False
    model_dim = 64
    model_states = 2
    model_dim_output = None # 0
    use_y = True
    use_yi = False
    steps = 1000
    keep_best = False
    num_threads = None # 50
    use_nearest_neighbor = False

    # WARNING: The following includes a curvature estimate, which
    #          is NOT only using stochastic gradient descent.
    ONLY_SGD = dict(
        faster_rate = 1.0,
        slower_rate = 1.0,
        update_ratio_step = 0.0,
        step_factor = 0.001,
        step_mean_change = 0.1,
        step_curv_change = 0.01,
        keep_best = keep_best,
        basis_replacement = False,
    )

    settings = dict(
        seed=seed,
        early_stop = False,
        log_grad_norm_frequency = 1,
        rank_check_frequency = 10,
        **({"mdo":model_dim_output} if model_dim_output is not None else {}),
        ax_normalized = False,
        x_normalized = False,
        y_normalized = False,
        # keep_best = False,
        # **ONLY_SGD
    )

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
    sizes = np.ones(n*len(functions), dtype="int32") * d

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

    print()
    print("ax.shape:    ", ax.shape)
    print("axi.shape:   ", axi.shape)
    print("sizes.shape: ", sizes.shape)
    print("x.shape:     ", x.shape)
    print("xi.shape:    ", xi.shape)
    print("y.shape:     ", y.shape)
    print("yi.shape:    ", yi.shape)
    print()

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

    # Train a new model or load an existing one.
    if (new_model or (not os.path.exists('temp-model.json'))):
        # Initialize a new model.
        print("Fitting model..")
        m = AXY(
            ads=agg_dim,
            ans=agg_states,
            mds=model_dim,
            mns=model_states,
            num_threads=num_threads,
            **settings,
        )
        print("m: ", m)
        print("m.config: ", m.config)
        m.fit(
            ax=(ax.copy() if use_a else None),
            axi=(axi if use_a else None),
            sizes=(sizes if use_a else None),
            x=(x.copy() if use_x else None),
            xi=(xi if use_x else None),
            y=(y.copy() if use_y else None),
            yi=(yi if use_yi else None),
            steps=steps,
            # nm = n // 3,
        )
        # Save and load the model.
        m.save("temp-model.json")

    # Load the saved model.
    m = AXY()
    m.load("temp-model.json")
    # Remove the saved model if only new models are desired.
    if new_model:
        os.remove("temp-model.json")

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
            ax=(ax if use_a else None),
            axi=(axi if use_a else None),
            sizes=(sizes if use_a else None),
            x=(x if use_x else None),
            xi=(xi if use_x else None),
            embedding=True
        )
        print("embeddings.shape: ", embeddings.shape)
        tree = BallTree(embeddings, build=False)
        print("Tree:")
        print(tree)


    # Define a function that evaluates the model with some different presets.
    def fhat(x, f=functions[0].__name__, yii=None):
        x = np.asarray(x, dtype="float32").reshape((-1,d))
        n = x.shape[0]
        xi = np.asarray([f]*n, dtype=object).reshape((-1,1))
        ax = x.reshape((-1,1)).copy()
        axi = np.asarray(list(zip(
            [f for _ in range(n*d)],
            [i+1 for _ in range(n) for i in range(d)]
        )), dtype=object)
        sizes = np.ones(x.shape[0], dtype="int32") * d
        if (not use_a): ax = axi = sizes = None
        if (not use_x): x = xi = None
        if (not use_yi): yii = None
        if (not use_nearest_neighbor):
            if (yii is None): return m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)[:,0]
            else:             return [yi_values[yii][v] for v in m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi)[:,(y.shape[1] if use_y else 0)+yii]]
        else:
            # Use the tree to lookup the nearest neighbor.
            emb = m(ax=ax, axi=axi, sizes=sizes, x=x, xi=xi, embedding=True)
            i = tree(emb, return_distance=False)
            if (yii is None): return y[i,0]
            else:             return [yi_values[yii][v] for v in yi[i.flatten(),yii]]

    # Generate the plot of the results.
    print("Adding to plot..")
    p = Plot()

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
                       plot_points=3000) # , mode="markers", marker_size=2)
    if (use_yi):
        # Add the provided points.
        for i, (f, xf) in enumerate(zip(functions, points)):
            for j in range(2):
                p.add(f"xi={f.__name__} yi={j}", *xf.T, [yi_values[j][v] for v in yi[:,j]],
                      color=j, group=j, shade=True, marker_size=5)
        # Add the function approximations.
        for f in functions:
            for i in range(yi.shape[1]):
                p.add_func(f"xi={f.__name__}, yi={i}", lambda x: fhat(x, f=f.__name__, yii=i),
                           *x_min_max, opacity=0.8, plot_points=3000,
                           mode="markers", shade=True, marker_size=4)

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
