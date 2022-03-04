import os
import numpy as np
from tlux.approximate.base import Approximator

_this_dir = os.path.dirname(os.path.abspath(__file__))
_source_code = os.path.join(_this_dir, "apos.f90")

class APOS(Approximator):
    # Make the string function return the unpacked model.
    def __str__(self): return str(self.model_unpacked())


    # Initialize a new APOS model.
    def __init__(self, **kwargs):
        import fmodpy
        apos = fmodpy.fimport(_source_code, blas=True, omp=True, wrap=True,
                              verbose=False, output_dir=_this_dir)
        # Store the Fortran module as an attribute.
        self.APOS = apos.apos
        # Set defaults for standard internal parameters.
        self.config = None
        self.model = np.zeros(0, dtype="float32")
        self.x_mean = np.zeros(1, dtype="float32")
        self.x_var = np.ones(1, dtype="float32")
        self.y_mean = np.zeros(1, dtype="float32")
        self.y_var = np.ones(1, dtype="float32")
        self.record = np.zeros(0, dtype="float32")
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
        self.seed = kwargs.pop("seed", None)
        self.steps = kwargs.pop("steps", 1000)
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
            self.x_mean = np.zeros(mdi, dtype="float32")
            self.x_var = np.ones(mdi, dtype="float32")
            self.y_mean = np.zeros(mdo, dtype="float32")
            self.y_var = np.ones(mdo, dtype="float32")
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
    def _fit(self, x, y, xi=None, normalize_x=True, normalize_y=True,
             new_model=False, **kwargs):
        # If this model isn't initialized, then do that.
        if (self.config is None):
            self._init_model(di=x.shape[1], do=y.shape[1])
        # Get the number of steps for training.
        steps = kwargs.get("steps", self.steps)
        # If a random seed is provided, then only 2 threads can be used
        #  because nondeterministic behavior is exhibited otherwise.
        seed = kwargs.get("seed", self.seed)
        if (seed is not None):
            num_threads = kwargs.get("num_threads", self.config.num_threads)
            if ((num_threads is None) or (num_threads > 2)):
                import warnings
                warnings.warn("Seeding a APOS model will deterministically initialize weights, but num_threads > 2 will result in nondeterministic model updates.")
        # Verify the shape of the model matches the data.
        mn = x.shape[0]
        mdi = x.shape[1]
        mdo = y.shape[1]
        mne = 0
        mde = 0
        # Check that the categories meet expectations.
        if (xi is not None):
            xi = np.asarray(xi, dtype="int32", order="C").T
            minval = xi.min()
            maxval = xi.max()
            num_categories = int(maxval - minval + 1)
            mne = max(self.config.mne, num_categories)
            mde = self.config.mde
            if (mne > 0): kwargs.update({"mne":mne})
            if (mde > 0): kwargs.update({"mde":mde})
        # ------------------------------------------------------------
        # If the shape of the model does not match the data, reinitialize.
        if (new_model or (self.config is None)
            or (self.config.mdi != mdi+mde)
            or (self.config.mdo != mdo)
            or ((mne > 0) and (mde <= 0))):
            kwargs.update({"mdi":mdi, "mdo":mdo})
            self._init_model(**kwargs)
        # Set any configuration keyword arguments given at initialization.
        for n in ({n for (n,t) in self.config._fields_} & set(kwargs)):
            setattr(self.config, n, kwargs[n])
        # Verify that the necessary number of embeddings fit into the model.
        if (xi is not None):
            assert (minval == 1), f"Expected minimum category in 'xi' to be 1, received {minval}."
            assert (maxval <= self.config.mne), f"Expected largest category in 'xi' to be at most {self.config.mne}, received {maxval}."
        else:
            xi = np.zeros((mn,0), dtype="int32", order="C")
        an = 0
        ax = np.zeros((an,0), dtype="float32", order="C")
        axi = np.zeros((an,0), dtype="int32", order="C")
        sizes = np.zeros(0, dtype="int32", order="C")
        # ------------------------------------------------------------
        # Make sure the points are stored properly in memory.
        x = np.asarray(x, dtype="float32", order="C")
        y = np.asarray(y, dtype="float32", order="C")
        # ------------------------------------------------------------
        # Minimize the mean squared error.
        self.record = np.zeros((steps,2), dtype="float32", order="C")
        result = self.APOS.minimize_mse(self.config, self.model,
                                        y.T, x.T, xi.T, ax.T, axi.T, sizes,
                                        steps=steps, record=self.record.T)
        assert (result[-1] == 0), f"APOS.minimize_mse returned nonzero exit code {result[-1]}."


    # Make predictions for new data.
    def _predict(self, x=None, xi=None, ax=None, axi=None, sizes=None, **kwargs):
        # Evaluate the model at all data.
        assert ((x is not None) or (sizes is not None)), "One of 'x' or 'sizes' must not be None for evaluate."
        # Make sure the positional x values are in expected format.
        if (x is not None):
            x = np.asarray(x, dtype="float32", order="C")
            mn = x.shape[0]
        else:
            mn = sizes.shape
            x = np.zeros((mn,0), dtype="float32", order="C")
        # Check that the positional categories meet expectations.
        if (xi is not None):
            xi = np.asarray(xi, dtype="int32", order="C")
            minval = xi.min()
            maxval = xi.max()
            assert (0 <= minval <= self.config.mne), f"Expected minimum category in 'xi' to be 0 (for unknown) or 1, received {minval}."
            assert (0 <= maxval <= self.config.mne), f"Expected largest category in 'xi' to be at most {self.config.mne}, received {maxval}."
            assert (xi.shape[1] == x.shape[0]), f"Provided 'x' {x.shape} and 'xi' {xi.shape} do not match."
        else:
            xi = np.zeros((mn,0), dtype="int32", order="C")
        # Make sure the apositional x values are in expected format.
        if (ax is not None):
            ax = np.asarray(ax, dtype="float32", order="C")
            an = ax.shape[0]
        else:
            an = 0
            ax = np.zeros((an,0), dtype="float32", order="C")
        # Check that the apositional categories meet expectations.
        if (axi is not None):
            axi = np.asarray(axi, dtype="int32", order="C")
            minval = axi.min()
            maxval = axi.max()
            assert (0 <= minval <= self.config.ane), f"Expected minimum category in 'axi' to be 0 (for unknown) or 1, received {minval}."
            assert (0 <= maxval <= self.config.ane), f"Expected largest category in 'axi' to be at most {self.config.ane}, received {maxval}."
            assert (axi.shape[1] == ax.shape[0]), f"Provided 'ax' {ax.shape} and 'axi' {axi.shape} do not match."
        else:
            axi = np.zeros((an,0), dtype="int32", order="C")
        # Check that the apositional argument sizes are formatted correctly.
        if (sizes is not None):
            sizes = np.asarray(sizes, dtype="int32", order="C")
        else:
            sizes = np.zeros(an, dtype="int32", order="C")
        # Embed the inpputs into the purely positional form.
        ado = self.config.ado
        mdo = self.config.mdo
        y = np.zeros((mn, mdo), dtype="float32", order="C")
        xxi = np.zeros((mn, self.config.mdi), dtype="float32", order="C")
        axxi = np.zeros((an, self.config.adi), dtype="float32", order="C")
        m_states = np.zeros((mn, self.config.mds, 2), dtype="float32", order="F")
        a_states = np.zeros((an, self.config.ads, 2), dtype="float32", order="F")
        ay = np.zeros((ado, an), dtype="float32", order="F")
        # Call the unerlying library.
        info = self.APOS.check_shape(self.config, self.model, y.T, x.T, xi.T, ax.T, axi.T, sizes)
        assert (info == 0), f"APOS.evaluate returned nonzero exit code {info}."
        self.APOS.embed(self.config, self.model, x.T, xi.T, ax.T, axi.T, xxi.T, axxi.T)
        self.APOS.evaluate(self.config, self.model, y.T, xxi.T, axxi.T,
                           sizes, m_states, a_states, ay, **kwargs)
        # Denormalize the output values and return them.
        return (y * self.y_var) + self.y_mean


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
            if (type(value) is list): 
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

    # Define a wrapper convenience function for computing principal components.
    from sklearn.decomposition import PCA        
    def pca(x, num_components=None):
        pca = PCA(n_components=num_components)
        if (num_components is None): num_components = min(*x.shape)
        else: num_components = min(num_components, *x.shape)
        pca.fit(x)
        return pca.components_, pca.singular_values_

    # A function for testing approximation algorithms.
    def f(x):
        x = x.reshape((-1,2))
        x, y = x[:,0], x[:,1]
        return 3*x + np.cos(8*x)/2 + np.sin(5*y)

    # TODO:
    #  - bounds on curvature based on the layer in the model
    #  - way to increase basis function diversity (different shift distribution?)
    #    it gets very low with depth of model using uniform and random ortho on sphere
    # 

    layer_dim = 32
    num_layers = 8
    np.random.seed(0)

    TEST_SAVE_LOAD = True
    TEST_INT_INPUT = False
    VISUALIZE_EMBEDDINGS = False
    TEST_SPECTRUM = False


    if TEST_SAVE_LOAD:
        # Try saving an untrained model.
        m = APOS()
        print("Empty model:")
        print(str(m))
        print()
        m.save("testing_empty_save.json")
        m.load("testing_empty_save.json")
        from util.approximate import PLRM
        m = APOS(mdi=2, mds=layer_dim, mns=num_layers, mdo=1, seed=0,
                 num_threads=1, steps=1000, 
                 ) # discontinuity=-1000.0) # initial_step=0.01)
        m2 = PLRM(di=2, ds=layer_dim, ns=num_layers, do=1, seed=0, num_threads=1, steps=1000)
        m2.model_unpacked().input_vecs[:,:] = m.model_unpacked().m_input_vecs[:,:]
        m2.model_unpacked().input_shift[:] = m.model_unpacked().m_input_shift[:]
        m2.model_unpacked().output_vecs[:,:] = m.model_unpacked().m_output_vecs[:,:]
        print(m)
        print(m2)
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
        print(m)
        print()
        m2.fit(x, y, normalize_x=False, normalize_y=False)
        # Add the data and the surface of the model to the plot.
        p = Plot()
        p.add("Data", *x.T, y)
        p.add_func("Fit", m, *x_min_max, vectorized=True)
        # Try saving the trained model and applying it after loading.
        print("Saving model:")
        print(m)
        # print(str(m)[:str(m).index("\n\n")])
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
            p.add("Step sizes", list(range(record.shape[0])), record[:,1], color=2, mode="lines")
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
        m = APOS(mdi=2, mds=layer_dim, mns=num_layers, mdo=1, mne=2, seed=0, steps=1000, num_threads=2)
        all_x = np.concatenate((x, x), axis=0)
        all_y = np.concatenate((y, np.cos(np.linalg.norm(x,axis=1))), axis=0)
        all_xi = np.concatenate((np.ones(len(x)),2*np.ones(len(x)))).reshape((-1,1)).astype("int32")
        m.fit(all_x, all_y, xi=all_xi)
        # Create an evaluation set that evaluates the model that was built over two differnt functions.
        xi1 = np.ones((len(x),1),dtype="int32")
        y1 = m(x, xi=xi1)
        y2 = m(x, xi=2*xi1)
        print("Adding to plot..")
        p = Plot()
        p.add("xi=1 true", *x.T, all_y[:len(all_y)//2], color=0)
        p.add("xi=2 true", *x.T, all_y[len(all_y)//2:], color=1)
        p.add_func("xi=1", lambda x: m(x, np.ones(len(x), dtype="int32").reshape((1,-1))), [0,1], [0,1], color=3, mode="markers", shade=True)
        p.add_func("xi=2", lambda x: m(x, 2*np.ones(len(x), dtype="int32").reshape((1,-1))), [0,1], [0,1], color=2, mode="markers", shade=True)

        # Generate the visual.
        print("Generating surface plot..")
        p.show(show=False)
        print("Generating loss plot..")
        p = type(p)("Mean squared error")
        # Rescale the columns of the record for visualization.
        record = m.record
        p.add("MSE", list(range(record.shape[0])), record[:,0], color=1, mode="lines")
        p.add("Step sizes", list(range(record.shape[0])), record[:,1], color=2, mode="lines")
        p.show(append=True, show=True)
        print("", "done.", flush=True)


    if TEST_SPECTRUM:
        n = 1000
        d_in = 3
        d_out = 5
        # Control array print options for readability.
        np.set_printoptions(edgeitems=50, linewidth=100000, 
            formatter=dict(float=lambda v: f"{v:.2f}"))

        # Plot the values (project onto 3D if there are more).
        visualize = False
        def plot_vals(x, name="", d=3, show=False):
            if (not visualize): return
            if (x.shape[1] > d):
                vecs, mags = pca(x, num_components=d)
                x = np.matmul(x, vecs.T)
            # Create the plot.
            p = Plot(name)
            p.add("", *x.T, color=1, marker_size=4, marker_line_width=1)
            p.plot(show=show, append=True, show_legend=False)
        # Initialize a model and data.
        m = APOS(mdi=d_in, mds=layer_dim, mns=num_layers, mdo=d_out, initial_shift_range=2.0)
        x = well_spaced_ball(n, d_in)
        y = np.vstack([
            np.cos(2*np.pi*(i+1)*np.linalg.norm(x,axis=1))
            for i in range(d_out)]).T
        # Show the input points.
        plot_vals(x, "Input")
        # Generate embeddings for all points.
        m(x, embeddings=True)
        for l in range(num_layers):
            emb = m.embeddings[:,:,l]
            print(l)
            print("  min", emb.min().round(2))#, emb.min(axis=0))
            print("  max", emb.max().round(2))#, emb.max(axis=0))
            print("  point spectrum:", pca(emb)[1])
            print("  basis spectrum:", pca(emb.T)[1])
            print()
            plot_vals(emb, f"Layer {l+1}", show=(l==num_layers-1))
        # Fit the model, then look at the new spectrum.
        print('-'*70)
        m.fit(x, y, steps=2000)
        m(x, embeddings=True)
        print('-'*70)
        #Show the spectrum.
        for l in range(num_layers):
            emb = m.embeddings[:,:,l]
            print(l)
            print("  min", emb.min().round(2))#, emb.min(axis=0))
            print("  max", emb.max().round(2))#, emb.max(axis=0))
            print("  point spectrum:", pca(emb)[1])
            print("  basis spectrum:", pca(emb.T)[1])
            print()
            plot_vals(emb, f"Fit Layer {l+1}", show=(l==num_layers-1))


    if VISUALIZE_EMBEDDINGS:
        # Create a blank-slate model, to see what it looks like.
        n = APOS(di=2, ds=layer_dim, ns=num_layers, do=1, seed=0, num_threads=2)
        from util.random import latin
        nx = latin(2000, 2)
        #   visialize the initial model at all layers.
        n(nx, embeddings=True, positions=True)
        embeddings = n.embeddings
        for i in range(num_layers):
            p = type(p)(f"Layer {i+1} initializations")
            p.add("Data", *x.T, y / 4, shade=True)
            for j in range(layer_dim):
                vals = embeddings[:,j,i]
                vals -= vals.min()
                if (vals.max() > 0):
                    vals /= vals.max()
                else:
                    vals -= .1
                p.add(f"Component {j+1}", *nx.T, vals, marker_size=3, use_gradient=True)
            p.show(append=(i>0), show=(i in {0, num_layers-1}))

        #   visualize the trained model at all layers
        n.fit(x, y)
        n(nx, embeddings=True, positions=True)
        embeddings = n.embeddings
        for i in range(num_layers):
            p = type(p)(f"Layer {i+1} final form")
            p.add("Data", *x.T, y / 4, shade=True)
            for j in range(layer_dim):
                vals = embeddings[:,j,i]
                vals -= vals.min()
                if (vals.max() > 0):
                    vals /= vals.max()
                else:
                    vals -= .1
                p.add(f"Component {j+1}", *nx.T, vals, marker_size=3, use_gradient=True)
            p.show(append=(i>0), show=(i in {0, num_layers-1}))

