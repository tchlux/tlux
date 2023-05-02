

# Build a class that contains pointers to the model internals, allowing
#  python attribute access to all of the different components of the models.
class AxyModel:
    def __init__(self, config, model, clock_rate=2000000000, show_vecs=False, show_times=True):
        self.config = config
        self.model = model
        self.clock_rate = clock_rate
        self.show_vecs = show_vecs
        self.show_times = show_times
        self.ax_shift      = self.model[self.config.aiss-1:self.config.aise].reshape(self.config.adn, order="F")
        self.ax_rescale    = self.model[self.config.aims-1:self.config.aime].reshape(self.config.adn, self.config.adn, order="F")
        self.a_embeddings  = self.model[self.config.asev-1:self.config.aeev].reshape(self.config.ade, self.config.ane, order="F")
        self.a_input_vecs  = self.model[self.config.asiv-1:self.config.aeiv].reshape(self.config.adi, self.config.ads, order="F")
        self.a_input_shift = self.model[self.config.asis-1:self.config.aeis].reshape(self.config.ads, order="F")
        self.a_state_vecs  = self.model[self.config.assv-1:self.config.aesv].reshape(self.config.ads, self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_state_shift = self.model[self.config.asss-1:self.config.aess].reshape(self.config.ads, max(0,self.config.ans-1), order="F")
        self.a_output_vecs = self.model[self.config.asov-1:self.config.aeov].reshape(self.config.adso, self.config.ado+1, order="F")
        self.ay_shift      = self.model[self.config.aoss-1:self.config.aose].reshape(self.config.ado, order="F")
        self.x_shift       = self.model[self.config.miss-1:self.config.mise].reshape(self.config.mdn, order="F")
        self.x_rescale     = self.model[self.config.mims-1:self.config.mime].reshape(self.config.mdn, self.config.mdn, order="F")
        self.m_embeddings  = self.model[self.config.msev-1:self.config.meev].reshape(self.config.mde, self.config.mne, order="F")
        self.m_input_vecs  = self.model[self.config.msiv-1:self.config.meiv].reshape(self.config.mdi, self.config.mds, order="F")
        self.m_input_shift = self.model[self.config.msis-1:self.config.meis].reshape(self.config.mds, order="F")
        self.m_state_vecs  = self.model[self.config.mssv-1:self.config.mesv].reshape(self.config.mds, self.config.mds, max(0,self.config.mns-1), order="F")
        self.m_state_shift = self.model[self.config.msss-1:self.config.mess].reshape(self.config.mds, max(0,self.config.mns-1), order="F")
        self.m_output_vecs = self.model[self.config.msov-1:self.config.meov].reshape(self.config.mdso, self.config.mdo, order="F")
        self.y_shift       = self.model[self.config.moss-1:self.config.mose].reshape(self.config.do, order="F")
        self.y_rescale     = self.model[self.config.moms-1:self.config.mome].reshape(self.config.do, self.config.do, order="F")

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
    def __str__(self, show_vecs=False, show_times=False):
        show_vecs = show_vecs or self.show_vecs
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
        if show_vecs: to_str = lambda arr: ("\n    " + "\n    ".join(str(arr).split("\n")) + "\n" if (arr.size > 0) else "\n")
        else:         to_str = lambda arr: "\n"
        # Provide details (and some values where possible).
        return (
            f"AXY model ({self.config.num_vars} variables, {self.config.total_size-self.config.num_vars} parameters) [{byte_size}]\n"+
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
        ) + (self._timer_summary_string() if (self.show_times or show_times) else "")


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



# Holder for the model and its work space (with named attributes).
class Details(dict):
    def __init__(self, config, steps=10, ydi=0, ywd=0,
                 model=None, rwork=None, iwork=None, lwork=None,
                 agg_iterators=None, record=None, yw=None, yi=None):
        import numpy as np
        self.config = config
        self.steps = steps
        # Generic allocations and objects.
        ftype = dict(order="F", dtype="float32")
        itype = dict(order="F", dtype="int32")
        ltype = dict(order="F", dtype="int64")
        if (model is None):
            model = np.ones(config.total_size, **ftype)
        if (rwork is None):
            rwork = np.ones(config.rwork_size, **ftype)  # beware of allocation, heap vs stack
        if (iwork is None):
            iwork = np.ones(config.iwork_size, **itype)
        if (lwork is None):
            lwork = np.ones(config.lwork_size, **ltype)
        if (agg_iterators is None):
            agg_iterators = np.ones((5, config.nmt), **ltype)
        if (record is None):
            record = np.zeros((6,steps), **ftype)
        if (yw is None):
            yw = np.zeros((ywd, config.nm), **ftype)
        if (yi is None):
            yi = np.zeros((ydi, config.nm), **ltype)
        # Store source memory allocations internally.
        self.model = model
        self.rwork = rwork
        self.iwork = iwork
        self.lwork = lwork
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
            x_shift = model[config.miss-1:config.mise].reshape(config.mdn, order="F") if (config.mdn > 0) else None,
            x_rescale = model[config.mims-1:config.mime].reshape(config.mdn, config.mdn, order="F"),
            y_shift = model[config.moss-1:config.mose].reshape(config.do, order="F"),
            y_rescale = model[config.moms-1:config.mome].reshape(config.do, config.do, order="F"),
            # Real work space.
            model_grad = rwork[config.smg-1:config.emg].reshape(config.num_vars, config.num_threads, order="F"),
            model_grad_mean = rwork[config.smgm-1:config.emgm].reshape(config.num_vars, order="F"),
            model_grad_curv = rwork[config.smgc-1:config.emgc].reshape(config.num_vars, order="F"),
            best_model = rwork[config.sbm-1:config.ebm].reshape(config.total_size, order="F"),
            ax = rwork[config.saxb-1:config.eaxb].reshape(config.adi, config.na, order="F"),
            a_emb_temp = rwork[config.saet-1:config.eaet].reshape(config.ade, config.ane, config.num_threads, order="F"),
            a_states = rwork[config.saxs-1:config.eaxs].reshape(config.na, config.ads, config.ans+1, order="F"),
            a_grads = rwork[config.saxg-1:config.eaxg].reshape(config.na, config.ads, config.ans+1, order="F"),
            ay = rwork[config.say-1:config.eay].reshape(config.na, config.ado+1, order="F"),
            ay_gradient = rwork[config.sayg-1:config.eayg].reshape(config.na, config.ado+1, order="F"),
            x = rwork[config.smxb-1:config.emxb].reshape(config.mdi, config.nm, order="F"),
            m_emb_temp = rwork[config.smet-1:config.emet].reshape(config.mde, config.mne, config.num_threads, order="F"),
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
            axi = lwork[config.saxi-1:config.eaxi].reshape(-1, config.na, order="F") if (config.saxi < config.eaxi) else np.zeros((0,0), dtype=int),
            sizes = lwork[config.ssb-1:config.esb].reshape(config.nm, order="F") if (config.ssb <= config.esb) else np.zeros(0, dtype=int),
            xi = lwork[config.smxi-1:config.emxi].reshape(-1, config.nm, order="F"),
            a_order = iwork[config.sao-1:config.eao].reshape(config.ads, config.num_threads, order="F"),
            m_order = iwork[config.smo-1:config.emo].reshape(config.mds, config.num_threads, order="F"),
            update_indices = lwork[config.sui-1:config.eui].reshape(config.num_vars, order="F"),
            # External space.
            yi = yi,
            yw = yw,
            record = record,
            agg_iterators = agg_iterators,
        ))

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)
    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)
