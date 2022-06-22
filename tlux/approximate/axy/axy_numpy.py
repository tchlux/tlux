# -----------------------------------------------------------------------
# License: MIT
#  Author: Thomas C.H. Lux
#  Source: https://github.com/tchlux/tlux
#    Date: 2022/04
# 
# Description:
#   A class that implements a numpy-only evaluation mode for AXY models.
# 
# -----------------------------------------------------------------------


import numpy as np


# Build a class that contains pointers to the model internals, allowing
#  python attribute access to all of the different components of the models.
#  Define a "predict" method that evaluates the model using only numpy
#  the same way as would be done in the compiled code.
class AxyNumpy:
    # Static method for loading a saved model from a path.
    def load(path):
        # Read the file.
        import json
        with open(path, "r") as f:
            attrs = json.loads(f.read())
        # Load the attributes of the model.
        for key in attrs:
            value = attrs[key]
            if (key[-4:] == "_map"):
                value = [np.asarray(l) for l in value]
            elif (key[:3] in {"xi_","yi_"}) or (key[:4] == "axi_"):
                pass
            elif (type(value) is list): 
                value = np.asarray(value, dtype="float32")
            attrs[key] = value
        # Return an initialized model.
        return AxyNumpy(config=attrs.pop("config",None), model=attrs.pop("model", None), **attrs)


    # Given a configuration object and a model array, store internal parameters.
    def __init__(self, config=None, model=None, **attrs):
        self.config = config
        self.model = model
        for attr in attrs: setattr(self, attr, attrs[attr])
        # Store the attributes of the model in self by breaking out
        #  the individual components 
        if (self.config is not None) and (self.model is not None):
            self.a_embeddings  = self.model[self.asev-1:self.aeev].reshape(self.ade, self.ane, order="F")
            self.a_input_vecs  = self.model[self.asiv-1:self.aeiv].reshape(self.adi, self.ads, order="F")
            self.a_input_shift = self.model[self.asis-1:self.aeis].reshape(self.ads, order="F")
            self.a_state_vecs  = self.model[self.assv-1:self.aesv].reshape(self.ads, self.ads, max(0,self.ans-1), order="F")
            self.a_state_shift = self.model[self.asss-1:self.aess].reshape(self.ads, max(0,self.ans-1), order="F")
            self.a_output_vecs = self.model[self.asov-1:self.aeov].reshape(self.adso, self.ado, order="F")
            self.m_embeddings  = self.model[self.msev-1:self.meev].reshape(self.mde, self.mne, order="F")
            self.m_input_vecs  = self.model[self.msiv-1:self.meiv].reshape(self.mdi, self.mds, order="F")
            self.m_input_shift = self.model[self.msis-1:self.meis].reshape(self.mds, order="F")
            self.m_state_vecs  = self.model[self.mssv-1:self.mesv].reshape(self.mds, self.mds, max(0,self.mns-1), order="F")
            self.m_state_shift = self.model[self.msss-1:self.mess].reshape(self.mds, max(0,self.mns-1), order="F")
            self.m_output_vecs = self.model[self.msov-1:self.meov].reshape(self.mdso, self.mdo, order="F")
            self.ax_shift = self.model[self.aiss-1:self.aise]
            self.ay_shift = self.model[self.aoss-1:self.aose]
            self.x_shift = self.model[self.miss-1:self.mise]
            self.y_shift = self.model[self.moss-1:self.mose]


    # Allow square brackets to access attributes of this model and its configuration.
    def __getattr__(self, attr):
        if (attr in self.config):
            return self.config[attr]
        else:
            return self.__getattribute__(attr)


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
        byte_size = len(self.config)*4 + self.model.dtype.itemsize*self.model.size
        byte_size = _byte_str(byte_size)
        # Create a function that prints the actual contents of the arrays.
        if vecs: to_str = lambda arr: "\n    " + "\n    ".join(str(arr).split("\n")) + "\n"
        else:    to_str = lambda arr: "\n"
        # Provide details (and some values where possible).
        return (
            f"AXY model ({self.total_size} parameters) [{byte_size}]\n"+
            (" aggregator model\n"+
            f"  input dimension  {self.adn}\n"+
            f"  output dimension {self.ado}\n"+
            f"  state dimension  {self.ads}\n"+
            f"  number of states {self.ans}\n"+
           (f"  embedding dimension  {self.ade}\n"+
            f"  number of embeddings {self.ane}\n"
             if self.ane > 0 else "")+
            f"  embeddings   {self.a_embeddings.shape}  "+to_str(self.a_embeddings)+
            f"  input vecs   {self.a_input_vecs.shape}  "+to_str(self.a_input_vecs)+
            f"  input shift  {self.a_input_shift.shape} "+to_str(self.a_input_shift)+
            f"  state vecs   {self.a_state_vecs.shape}  "+to_str(self.a_state_vecs)+
            f"  state shift  {self.a_state_shift.shape} "+to_str(self.a_state_shift)+
            f"  output vecs  {self.a_output_vecs.shape} "+to_str(self.a_output_vecs)+
             "\n" if (self.a_output_vecs.size > 0) else "") +
            (" model\n"+
            f"  input dimension  {self.mdn}\n"+
            f"  output dimension {self.mdo}\n"+
            f"  state dimension  {self.mds}\n"+
            f"  number of states {self.mns}\n"+
           (f"  embedding dimension  {self.mde}\n"+
            f"  number of embeddings {self.mne}\n"
             if self.mne > 0 else "")+
            f"  embeddings   {self.m_embeddings.shape}  "+to_str(self.m_embeddings)+
            f"  input vecs   {self.m_input_vecs.shape}  "+to_str(self.m_input_vecs)+
            f"  input shift  {self.m_input_shift.shape} "+to_str(self.m_input_shift)+
            f"  state vecs   {self.m_state_vecs.shape}  "+to_str(self.m_state_vecs)+
            f"  state shift  {self.m_state_shift.shape} "+to_str(self.m_state_shift)+
            f"  output vecs  {self.m_output_vecs.shape} "+to_str(self.m_output_vecs)
             if (self.m_output_vecs.size > 0) else "")
        )


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


    # Calling this model is an alias for 'AXY.predict'.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    # Make predictions for new data.
    def predict(self, x=None, xi=None, ax=None, axi=None, sizes=None):
        # Evaluate the model at all data.
        assert ((x is not None) or (xi is not None) or (sizes is not None)), "AXY.predict requires at least one of 'x', 'xi', or 'sizes' to not be None."
        # Make sure that 'sizes' were provided for aggregator (aggregate) inputs.
        if ((ax is not None) or (axi is not None)):
            assert (sizes is not None), "AXY.predict requires 'sizes' to be provided for aggregator input sets (ax and axi)."
        # Make sure that all inputs are numpy arrays.
        nm, na, mdn, mne, mdo, adn, ane, yne, _, x, xi, ax, axi, sizes = (
            self._to_array(ax, axi, sizes, x, xi, None, None)
        )
        # Embed the inputs into the purely positional form.
        ade = self.ade
        ads = self.ads
        ado = self.ado
        mde = self.mde
        mds = self.mds
        if (self.mdo != 0):
            mdo = self.mdo
        else:
            mdo = self.ado
        # Compute the true real-vector input dimensions given embeddings.
        adn += ade
        mdn += mde + ado
        # ------------------------------------------------------------
        # Initialize storage for all arrays needed at evaluation time.
        #   If there are integer embeddings, expand "ax" and "x" to have
        #   space to hold those embeddings.
        if (self.ade > 0):
            _ax = np.zeros((ax.shape[0],ax.shape[1]+self.ade), dtype="float32", order="C")
            _ax[:,:ax.shape[1]] = ax
            ax = _ax
        ay = np.zeros((na, ado), dtype="float32", order="F")
        if (self.mde > 0) or (self.ado > 0):
            _x = np.zeros((x.shape[0],self.mdi), dtype="float32", order="C")
            _x[:,:x.shape[1]] = x
            x = _x
        y = np.zeros((nm, mdo), dtype="float32", order="C")
        # ------------------------------------------------------------
        # Perform all data and array validation checks to make sure sizes are correct.
        assert (x.shape[0] == y.shape[0]), f"x {x.shape} and y {y.shape} do not have the same numer of rows."
        assert (x.shape[1] == self.mdi), f"x {x.shape} must have {self.mdi} columns."
        assert ((self.mdo == 0) or (self.mdo == y.shape[1])), f"y {y.shape} must have {self.mdo} columns."
        assert ((self.mdo > 0) or (self.ado == y.shape[1])), f"y {y.shape} must have {self.ado} columns."
        assert ((self.mne == 0) or (xi.shape[0] == x.shape[0])), f"xi {xi.shape} must have same number of rows as x {x.shape}."
        if (xi.size > 0):
            assert ((xi.min() >= 0) and (xi.max() <= self.mne)), f"xi [{xi.min()},{xi.max()}] must be in range [0,{self.mne}]."
        assert ((self.adi == 0) or (sizes.size == y.shape[0])), f"sizes {sizes.size} must have size equal to number of rows in y {y.shape}."
        assert (sizes.sum() == ax.shape[0]), f"sum(sizes) = {sizes.sum()} but should equal number of rows in ax {ax.shape}."
        assert (self.adi == ax.shape[1]), f"ax {ax.shape} must have {self.adi} columns."
        assert (ax.shape[0] == axi.shape[0]), f"ax {ax.shape} number of rows does not match axi {axi.shape}."
        if (axi.size > 0):
            assert ((axi.min() >= 0) and (axi.max() <= self.ane)), f"axi [{axi.min()},{axi.max()}] must be in range [0,{self.ane}]."
        # ------------------------------------------------------------
        # Perform the embedding operation, putting all embedding vectors into input format.
        if (self.ade > 0):
            a_known = axi > 0
            a_embeddings = (
                (self.a_embeddings[:,axi.flatten()-1].reshape((-1,)+axi.shape) # get all embedding vectors (ade, an, axi.shape[1])
                 * a_known) # zero out the "unkown" embeddings (integer value = 0)
                .sum(axis=-1) / np.maximum(1,a_known.sum(axis=-1)) # take mean of however many embeddings were known (count > 0)
            )
            ax[:, self.adn:self.adn+self.ade] = a_embeddings.T
        if (self.mde > 0):
            m_known = xi > 0
            m_embeddings = (
                (self.m_embeddings[:,xi.flatten()-1].reshape((-1,)+xi.shape) # get all embedding vectors (mde, mn, xi.shape[1])
                 * m_known) # zero out the "unkown" embeddings (integer value = 0)
                .sum(axis=-1) / np.maximum(1,m_known.sum(axis=-1)) # take mean of however many embeddings were known (count > 0)
            )
            x[:, self.mdn:self.mdn+self.mde] = m_embeddings.T
        # ------------------------------------------------------------
        # Evaluate aggregator model first, if it exists.
        if (self.adi > 0):
            state = ax
            state[:,:self.adn] += self.ax_shift[:]
            if (self.ans > 0):
                state = np.maximum(self.discontinuity, self.a_input_shift
                                   + np.matmul(state, self.a_input_vecs))
                for i in range(self.ans - 1):
                    state = np.maximum(self.discontinuity, self.a_state_shift[:,i]
                                       + np.matmul(state, self.a_state_vecs[:,:,i]))
            state = np.matmul(state, self.a_output_vecs)
            # Aggregate over set outputs.
            if (self.mdo > 0):
                ay_start = self.mdn + self.mde
                set_start = set_end = 0
                for i in range(sizes.shape[0]):
                    set_end += sizes[i]
                    x[i,ay_start:] = state[set_start:set_end].mean(axis=0)
                    set_start = set_end
                x[:,ay_start:] += self.ay_shift[:]
            else:
                set_start = set_end = 0
                for i in range(sizes.shape[0]):
                    set_end += sizes[i]
                    y[i,:] = state[set_start:set_end].mean(axis=0)
                    set_start = set_end
        # Evaluate the positional model second, if it exists.
        if (self.mdo > 0):
            state = x
            state[:,:self.mdn] += self.x_shift[:]
            if (self.mns > 0):
                state = np.maximum(self.discontinuity, self.m_input_shift
                                   + np.matmul(state, self.m_input_vecs))
                for i in range(self.mns - 1):
                    state = np.maximum(self.discontinuity, self.m_state_shift[:,i]
                                       + np.matmul(state, self.m_state_vecs[:,:,i]))
            y[:,:] = np.matmul(state, self.m_output_vecs)            
        # Add output shift.
        y[:,:] += self.y_shift[:]
        # ------------------------------------------------------------
        # If there are embedded y values in the output, return them to the format at training time.
        if (len(self.yi_map) > 0):
            yne = sum(self.yi_sizes)
            _y = [y[:,i] for i in range(y.shape[1]-yne)]
            for i in range(len(self.yi_map)):
                start = self.yi_starts[i]
                size = self.yi_sizes[i]
                _y.append(
                    self.yi_map[i][np.argmax(y[:,start:start+size], axis=1)]
                )
            return np.asarray(_y).T
        else:
            return y
