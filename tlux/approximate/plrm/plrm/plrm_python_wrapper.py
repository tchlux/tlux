'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

! TODO:
!
! - Add the zero mean and unit length code into the `MINIMIZE_MSE` routine
!   then embed those shifts into the first layer of the model at training end.
!
! - Add optional rank normalization to model input and output, with a
!   fixed number of rank positions (100). Include index sorting code. Use
!   binary search mixed with walking sorted to transform new points.
!
! - Add well spaced sphere design code, replace random sphere usage
!   with the well spaced sphere. Especially for embedding initialization.
!
! - Zero mean and unit variance the input and output data inside the
!   fit routine, then insert those linear operators into the weights
!   of the model (to reduce work that needs to be done outside).
!
! - Get stats on the internal values within the network during training.
!   - step size progression
!   - shift values
!   - vector magnitudes for each node
!   - output weights magnitude for each node
!   - internal node contributions to MSE
!   - data distributions at internal nodes (% less and greater than 0)
!
! - Implement apositional-positional approximation that stacks a
!   mean aggregation model in front of a standard model.
!
! - Training support for vector of assigned neighbors for each point.
!
! - Train only the output, internal, or input layers.
!


! Module for matrix multiplication (absolutely crucial for PLRM speed).
'''

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "plrm." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3', '-lblas', '-fopenmp']
_ordered_dependencies = ['plrm.f90', 'plrm_c_wrapper.f90']
# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the existing object. If that fails, recompile and then try.
try:
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = " ".join([_fort_compiler] + _compile_options + ["-o", _shared_object_name] + _ordered_dependencies)
    if _verbose:
        print("Running system command with arguments")
        print("  ", _command)
    # Run the compilation command.
    import subprocess
    subprocess.run(_command, shell=True, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


class matrix_multiplication:
    ''''''

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine GEMM
    
    def gemm(self, op_a, op_b, out_rows, out_cols, inner_dim, ab_mult, a, a_rows, b, b_rows, c_mult, c, c_rows):
        '''! Convenience wrapper routine for calling matrix multiply.'''
        
        # Setting up "op_a"
        if (type(op_a) is not ctypes.c_char): op_a = ctypes.c_char(op_a)
        
        # Setting up "op_b"
        if (type(op_b) is not ctypes.c_char): op_b = ctypes.c_char(op_b)
        
        # Setting up "out_rows"
        if (type(out_rows) is not ctypes.c_int): out_rows = ctypes.c_int(out_rows)
        
        # Setting up "out_cols"
        if (type(out_cols) is not ctypes.c_int): out_cols = ctypes.c_int(out_cols)
        
        # Setting up "inner_dim"
        if (type(inner_dim) is not ctypes.c_int): inner_dim = ctypes.c_int(inner_dim)
        
        # Setting up "ab_mult"
        if (type(ab_mult) is not ctypes.c_float): ab_mult = ctypes.c_float(ab_mult)
        
        # Setting up "a"
        if ((not issubclass(type(a), numpy.ndarray)) or
            (not numpy.asarray(a).flags.f_contiguous) or
            (not (a.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            a = numpy.asarray(a, dtype=ctypes.c_float, order='F')
        a_dim_1 = ctypes.c_int(a.shape[0])
        a_dim_2 = ctypes.c_int(a.shape[1])
        
        # Setting up "a_rows"
        if (type(a_rows) is not ctypes.c_int): a_rows = ctypes.c_int(a_rows)
        
        # Setting up "b"
        if ((not issubclass(type(b), numpy.ndarray)) or
            (not numpy.asarray(b).flags.f_contiguous) or
            (not (b.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'b' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            b = numpy.asarray(b, dtype=ctypes.c_float, order='F')
        b_dim_1 = ctypes.c_int(b.shape[0])
        b_dim_2 = ctypes.c_int(b.shape[1])
        
        # Setting up "b_rows"
        if (type(b_rows) is not ctypes.c_int): b_rows = ctypes.c_int(b_rows)
        
        # Setting up "c_mult"
        if (type(c_mult) is not ctypes.c_float): c_mult = ctypes.c_float(c_mult)
        
        # Setting up "c"
        if ((not issubclass(type(c), numpy.ndarray)) or
            (not numpy.asarray(c).flags.f_contiguous) or
            (not (c.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'c' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            c = numpy.asarray(c, dtype=ctypes.c_float, order='F')
        c_dim_1 = ctypes.c_int(c.shape[0])
        c_dim_2 = ctypes.c_int(c.shape[1])
        
        # Setting up "c_rows"
        if (type(c_rows) is not ctypes.c_int): c_rows = ctypes.c_int(c_rows)
    
        # Call C-accessible Fortran wrapper.
        clib.c_gemm(ctypes.byref(op_a), ctypes.byref(op_b), ctypes.byref(out_rows), ctypes.byref(out_cols), ctypes.byref(inner_dim), ctypes.byref(ab_mult), ctypes.byref(a_dim_1), ctypes.byref(a_dim_2), ctypes.c_void_p(a.ctypes.data), ctypes.byref(a_rows), ctypes.byref(b_dim_1), ctypes.byref(b_dim_2), ctypes.c_void_p(b.ctypes.data), ctypes.byref(b_rows), ctypes.byref(c_mult), ctypes.byref(c_dim_1), ctypes.byref(c_dim_2), ctypes.c_void_p(c.ctypes.data), ctypes.byref(c_rows))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return c

matrix_multiplication = matrix_multiplication()


class plrm:
    '''! A piecewise linear regression model.'''
    
    # This defines a C structure that can be used to hold this defined type.
    class MODEL_CONFIG(ctypes.Structure):
        # (name, ctype) fields for this structure.
        _fields_ = [("mdi", ctypes.c_int), ("mds", ctypes.c_int), ("mns", ctypes.c_int), ("mdo", ctypes.c_int), ("mne", ctypes.c_int), ("mde", ctypes.c_int), ("total_size", ctypes.c_int), ("num_vars", ctypes.c_int), ("siv", ctypes.c_int), ("eiv", ctypes.c_int), ("sis", ctypes.c_int), ("eis", ctypes.c_int), ("sin", ctypes.c_int), ("ein", ctypes.c_int), ("six", ctypes.c_int), ("eix", ctypes.c_int), ("ssv", ctypes.c_int), ("esv", ctypes.c_int), ("sss", ctypes.c_int), ("ess", ctypes.c_int), ("ssn", ctypes.c_int), ("esn", ctypes.c_int), ("ssx", ctypes.c_int), ("esx", ctypes.c_int), ("sov", ctypes.c_int), ("eov", ctypes.c_int), ("sev", ctypes.c_int), ("eev", ctypes.c_int), ("discontinuity", ctypes.c_float), ("initial_shift_range", ctypes.c_float), ("initial_output_scale", ctypes.c_float), ("initial_step", ctypes.c_float), ("initial_step_mean_change", ctypes.c_float), ("initial_step_curv_change", ctypes.c_float), ("faster_rate", ctypes.c_float), ("slower_rate", ctypes.c_float), ("min_steps_to_stability", ctypes.c_int), ("num_threads", ctypes.c_int), ("keep_best", ctypes.c_bool), ("early_stop", ctypes.c_bool)]
        # Define an "__init__" that can take a class or keyword arguments as input.
        def __init__(self, value=0, **kwargs):
            # From whatever object (or dictionary) was given, assign internal values.
            self.mdi = kwargs.get("mdi", getattr(value, "mdi", value))
            self.mds = kwargs.get("mds", getattr(value, "mds", value))
            self.mns = kwargs.get("mns", getattr(value, "mns", value))
            self.mdo = kwargs.get("mdo", getattr(value, "mdo", value))
            self.mne = kwargs.get("mne", getattr(value, "mne", value))
            self.mde = kwargs.get("mde", getattr(value, "mde", value))
            self.total_size = kwargs.get("total_size", getattr(value, "total_size", value))
            self.num_vars = kwargs.get("num_vars", getattr(value, "num_vars", value))
            self.siv = kwargs.get("siv", getattr(value, "siv", value))
            self.eiv = kwargs.get("eiv", getattr(value, "eiv", value))
            self.sis = kwargs.get("sis", getattr(value, "sis", value))
            self.eis = kwargs.get("eis", getattr(value, "eis", value))
            self.sin = kwargs.get("sin", getattr(value, "sin", value))
            self.ein = kwargs.get("ein", getattr(value, "ein", value))
            self.six = kwargs.get("six", getattr(value, "six", value))
            self.eix = kwargs.get("eix", getattr(value, "eix", value))
            self.ssv = kwargs.get("ssv", getattr(value, "ssv", value))
            self.esv = kwargs.get("esv", getattr(value, "esv", value))
            self.sss = kwargs.get("sss", getattr(value, "sss", value))
            self.ess = kwargs.get("ess", getattr(value, "ess", value))
            self.ssn = kwargs.get("ssn", getattr(value, "ssn", value))
            self.esn = kwargs.get("esn", getattr(value, "esn", value))
            self.ssx = kwargs.get("ssx", getattr(value, "ssx", value))
            self.esx = kwargs.get("esx", getattr(value, "esx", value))
            self.sov = kwargs.get("sov", getattr(value, "sov", value))
            self.eov = kwargs.get("eov", getattr(value, "eov", value))
            self.sev = kwargs.get("sev", getattr(value, "sev", value))
            self.eev = kwargs.get("eev", getattr(value, "eev", value))
            self.discontinuity = kwargs.get("discontinuity", getattr(value, "discontinuity", value))
            self.initial_shift_range = kwargs.get("initial_shift_range", getattr(value, "initial_shift_range", value))
            self.initial_output_scale = kwargs.get("initial_output_scale", getattr(value, "initial_output_scale", value))
            self.initial_step = kwargs.get("initial_step", getattr(value, "initial_step", value))
            self.initial_step_mean_change = kwargs.get("initial_step_mean_change", getattr(value, "initial_step_mean_change", value))
            self.initial_step_curv_change = kwargs.get("initial_step_curv_change", getattr(value, "initial_step_curv_change", value))
            self.faster_rate = kwargs.get("faster_rate", getattr(value, "faster_rate", value))
            self.slower_rate = kwargs.get("slower_rate", getattr(value, "slower_rate", value))
            self.min_steps_to_stability = kwargs.get("min_steps_to_stability", getattr(value, "min_steps_to_stability", value))
            self.num_threads = kwargs.get("num_threads", getattr(value, "num_threads", value))
            self.keep_best = kwargs.get("keep_best", getattr(value, "keep_best", value))
            self.early_stop = kwargs.get("early_stop", getattr(value, "early_stop", value))
    

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine NEW_MODEL_CONFIG
    
    def new_model_config(self, mdi, mdo, mds=None, mns=None, mne=None, mde=None, num_threads=None):
        '''! Generate a model configuration given state parameters for the model.
! Size related parameters.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "mdi"
        if (type(mdi) is not ctypes.c_int): mdi = ctypes.c_int(mdi)
        
        # Setting up "mdo"
        if (type(mdo) is not ctypes.c_int): mdo = ctypes.c_int(mdo)
        
        # Setting up "mds"
        mds_present = ctypes.c_bool(True)
        if (mds is None):
            mds_present = ctypes.c_bool(False)
            mds = ctypes.c_int()
        if (type(mds) is not ctypes.c_int): mds = ctypes.c_int(mds)
        
        # Setting up "mns"
        mns_present = ctypes.c_bool(True)
        if (mns is None):
            mns_present = ctypes.c_bool(False)
            mns = ctypes.c_int()
        if (type(mns) is not ctypes.c_int): mns = ctypes.c_int(mns)
        
        # Setting up "mne"
        mne_present = ctypes.c_bool(True)
        if (mne is None):
            mne_present = ctypes.c_bool(False)
            mne = ctypes.c_int()
        if (type(mne) is not ctypes.c_int): mne = ctypes.c_int(mne)
        
        # Setting up "mde"
        mde_present = ctypes.c_bool(True)
        if (mde is None):
            mde_present = ctypes.c_bool(False)
            mde = ctypes.c_int()
        if (type(mde) is not ctypes.c_int): mde = ctypes.c_int(mde)
        
        # Setting up "num_threads"
        num_threads_present = ctypes.c_bool(True)
        if (num_threads is None):
            num_threads_present = ctypes.c_bool(False)
            num_threads = ctypes.c_int()
        if (type(num_threads) is not ctypes.c_int): num_threads = ctypes.c_int(num_threads)
        
        # Setting up "config"
        config = MODEL_CONFIG()
    
        # Call C-accessible Fortran wrapper.
        clib.c_new_model_config(ctypes.byref(mdi), ctypes.byref(mdo), ctypes.byref(mds_present), ctypes.byref(mds), ctypes.byref(mns_present), ctypes.byref(mns), ctypes.byref(mne_present), ctypes.byref(mne), ctypes.byref(mde_present), ctypes.byref(mde), ctypes.byref(num_threads_present), ctypes.byref(num_threads), ctypes.byref(config))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return config

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine INIT_MODEL
    
    def init_model(self, config, model, seed=None):
        '''! Initialize the weights for a model, optionally provide a random seed.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "model"
        if ((not issubclass(type(model), numpy.ndarray)) or
            (not numpy.asarray(model).flags.f_contiguous) or
            (not (model.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'model' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            model = numpy.asarray(model, dtype=ctypes.c_float, order='F')
        model_dim_1 = ctypes.c_int(model.shape[0])
        
        # Setting up "seed"
        seed_present = ctypes.c_bool(True)
        if (seed is None):
            seed_present = ctypes.c_bool(False)
            seed = ctypes.c_int()
        if (type(seed) is not ctypes.c_int): seed = ctypes.c_int(seed)
    
        # Call C-accessible Fortran wrapper.
        clib.c_init_model(ctypes.byref(config), ctypes.byref(model_dim_1), ctypes.c_void_p(model.ctypes.data), ctypes.byref(seed_present), ctypes.byref(seed))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return 

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine RANDOM_UNIT_VECTORS
    
    def random_unit_vectors(self, column_vectors):
        '''! Generate randomly distributed vectors on the N-sphere.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "column_vectors"
        if ((not issubclass(type(column_vectors), numpy.ndarray)) or
            (not numpy.asarray(column_vectors).flags.f_contiguous) or
            (not (column_vectors.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'column_vectors' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            column_vectors = numpy.asarray(column_vectors, dtype=ctypes.c_float, order='F')
        column_vectors_dim_1 = ctypes.c_int(column_vectors.shape[0])
        column_vectors_dim_2 = ctypes.c_int(column_vectors.shape[1])
    
        # Call C-accessible Fortran wrapper.
        clib.c_random_unit_vectors(ctypes.byref(column_vectors_dim_1), ctypes.byref(column_vectors_dim_2), ctypes.c_void_p(column_vectors.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return column_vectors

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine UNPACK_EMBEDDINGS
    
    def unpack_embeddings(self, config, embeddings, int_inputs, embedded):
        '''! Given integer inputs and embedding vectors, put embeddings in
!  place of integer inputs inside of a real matrix.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "embeddings"
        if ((not issubclass(type(embeddings), numpy.ndarray)) or
            (not numpy.asarray(embeddings).flags.f_contiguous) or
            (not (embeddings.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'embeddings' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            embeddings = numpy.asarray(embeddings, dtype=ctypes.c_float, order='F')
        embeddings_dim_1 = ctypes.c_int(embeddings.shape[0])
        embeddings_dim_2 = ctypes.c_int(embeddings.shape[1])
        
        # Setting up "int_inputs"
        if ((not issubclass(type(int_inputs), numpy.ndarray)) or
            (not numpy.asarray(int_inputs).flags.f_contiguous) or
            (not (int_inputs.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'int_inputs' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            int_inputs = numpy.asarray(int_inputs, dtype=ctypes.c_int, order='F')
        int_inputs_dim_1 = ctypes.c_int(int_inputs.shape[0])
        int_inputs_dim_2 = ctypes.c_int(int_inputs.shape[1])
        
        # Setting up "embedded"
        if ((not issubclass(type(embedded), numpy.ndarray)) or
            (not numpy.asarray(embedded).flags.f_contiguous) or
            (not (embedded.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'embedded' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            embedded = numpy.asarray(embedded, dtype=ctypes.c_float, order='F')
        embedded_dim_1 = ctypes.c_int(embedded.shape[0])
        embedded_dim_2 = ctypes.c_int(embedded.shape[1])
    
        # Call C-accessible Fortran wrapper.
        clib.c_unpack_embeddings(ctypes.byref(config), ctypes.byref(embeddings_dim_1), ctypes.byref(embeddings_dim_2), ctypes.c_void_p(embeddings.ctypes.data), ctypes.byref(int_inputs_dim_1), ctypes.byref(int_inputs_dim_2), ctypes.c_void_p(int_inputs.ctypes.data), ctypes.byref(embedded_dim_1), ctypes.byref(embedded_dim_2), ctypes.c_void_p(embedded.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return embedded

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine RESET_MIN_MAX
    
    def reset_min_max(self, input_min, input_max, state_min, state_max):
        '''! Reset the minimum and maximum values for internal nonlinearities.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "input_min"
        if ((not issubclass(type(input_min), numpy.ndarray)) or
            (not numpy.asarray(input_min).flags.f_contiguous) or
            (not (input_min.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'input_min' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            input_min = numpy.asarray(input_min, dtype=ctypes.c_float, order='F')
        input_min_dim_1 = ctypes.c_int(input_min.shape[0])
        
        # Setting up "input_max"
        if ((not issubclass(type(input_max), numpy.ndarray)) or
            (not numpy.asarray(input_max).flags.f_contiguous) or
            (not (input_max.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'input_max' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            input_max = numpy.asarray(input_max, dtype=ctypes.c_float, order='F')
        input_max_dim_1 = ctypes.c_int(input_max.shape[0])
        
        # Setting up "state_min"
        if ((not issubclass(type(state_min), numpy.ndarray)) or
            (not numpy.asarray(state_min).flags.f_contiguous) or
            (not (state_min.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'state_min' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            state_min = numpy.asarray(state_min, dtype=ctypes.c_float, order='F')
        state_min_dim_1 = ctypes.c_int(state_min.shape[0])
        state_min_dim_2 = ctypes.c_int(state_min.shape[1])
        
        # Setting up "state_max"
        if ((not issubclass(type(state_max), numpy.ndarray)) or
            (not numpy.asarray(state_max).flags.f_contiguous) or
            (not (state_max.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'state_max' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            state_max = numpy.asarray(state_max, dtype=ctypes.c_float, order='F')
        state_max_dim_1 = ctypes.c_int(state_max.shape[0])
        state_max_dim_2 = ctypes.c_int(state_max.shape[1])
    
        # Call C-accessible Fortran wrapper.
        clib.c_reset_min_max(ctypes.byref(input_min_dim_1), ctypes.c_void_p(input_min.ctypes.data), ctypes.byref(input_max_dim_1), ctypes.c_void_p(input_max.ctypes.data), ctypes.byref(state_min_dim_1), ctypes.byref(state_min_dim_2), ctypes.c_void_p(state_min.ctypes.data), ctypes.byref(state_max_dim_1), ctypes.byref(state_max_dim_2), ctypes.c_void_p(state_max.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return input_min, input_max, state_min, state_max

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine SET_MIN_MAX
    
    def set_min_max(self, config, model, x, xi=None):
        '''! Evaluate the piecewise linear regression model and store the minimum
!  and maximum values observed at each internal piecewise linear function.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "model"
        if ((not issubclass(type(model), numpy.ndarray)) or
            (not numpy.asarray(model).flags.f_contiguous) or
            (not (model.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'model' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            model = numpy.asarray(model, dtype=ctypes.c_float, order='F')
        model_dim_1 = ctypes.c_int(model.shape[0])
        
        # Setting up "x"
        if ((not issubclass(type(x), numpy.ndarray)) or
            (not numpy.asarray(x).flags.f_contiguous) or
            (not (x.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'x' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            x = numpy.asarray(x, dtype=ctypes.c_float, order='F')
        x_dim_1 = ctypes.c_int(x.shape[0])
        x_dim_2 = ctypes.c_int(x.shape[1])
        
        # Setting up "xi"
        xi_present = ctypes.c_bool(True)
        if (xi is None):
            xi_present = ctypes.c_bool(False)
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif (type(xi) == bool) and (xi):
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif ((not issubclass(type(xi), numpy.ndarray)) or
              (not numpy.asarray(xi).flags.f_contiguous) or
              (not (xi.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'xi' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            xi = numpy.asarray(xi, dtype=ctypes.c_int, order='F')
        if (xi_present):
            xi_dim_1 = ctypes.c_int(xi.shape[0])
            xi_dim_2 = ctypes.c_int(xi.shape[1])
        else:
            xi_dim_1 = ctypes.c_int()
            xi_dim_2 = ctypes.c_int()
    
        # Call C-accessible Fortran wrapper.
        clib.c_set_min_max(ctypes.byref(config), ctypes.byref(model_dim_1), ctypes.c_void_p(model.ctypes.data), ctypes.byref(x_dim_1), ctypes.byref(x_dim_2), ctypes.c_void_p(x.ctypes.data), ctypes.byref(xi_present), ctypes.byref(xi_dim_1), ctypes.byref(xi_dim_2), ctypes.c_void_p(xi.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return model

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine CHECK_SHAPE
    
    def check_shape(self, config, model, x, y, xi=None):
        '''! Returnn nonzero INFO if any shapes do not match expectations.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "model"
        if ((not issubclass(type(model), numpy.ndarray)) or
            (not numpy.asarray(model).flags.f_contiguous) or
            (not (model.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'model' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            model = numpy.asarray(model, dtype=ctypes.c_float, order='F')
        model_dim_1 = ctypes.c_int(model.shape[0])
        
        # Setting up "x"
        if ((not issubclass(type(x), numpy.ndarray)) or
            (not numpy.asarray(x).flags.f_contiguous) or
            (not (x.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'x' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            x = numpy.asarray(x, dtype=ctypes.c_float, order='F')
        x_dim_1 = ctypes.c_int(x.shape[0])
        x_dim_2 = ctypes.c_int(x.shape[1])
        
        # Setting up "y"
        if ((not issubclass(type(y), numpy.ndarray)) or
            (not numpy.asarray(y).flags.f_contiguous) or
            (not (y.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'y' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            y = numpy.asarray(y, dtype=ctypes.c_float, order='F')
        y_dim_1 = ctypes.c_int(y.shape[0])
        y_dim_2 = ctypes.c_int(y.shape[1])
        
        # Setting up "xi"
        xi_present = ctypes.c_bool(True)
        if (xi is None):
            xi_present = ctypes.c_bool(False)
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif (type(xi) == bool) and (xi):
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif ((not issubclass(type(xi), numpy.ndarray)) or
              (not numpy.asarray(xi).flags.f_contiguous) or
              (not (xi.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'xi' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            xi = numpy.asarray(xi, dtype=ctypes.c_int, order='F')
        if (xi_present):
            xi_dim_1 = ctypes.c_int(xi.shape[0])
            xi_dim_2 = ctypes.c_int(xi.shape[1])
        else:
            xi_dim_1 = ctypes.c_int()
            xi_dim_2 = ctypes.c_int()
        
        # Setting up "info"
        info = ctypes.c_int()
    
        # Call C-accessible Fortran wrapper.
        clib.c_check_shape(ctypes.byref(config), ctypes.byref(model_dim_1), ctypes.c_void_p(model.ctypes.data), ctypes.byref(x_dim_1), ctypes.byref(x_dim_2), ctypes.c_void_p(x.ctypes.data), ctypes.byref(y_dim_1), ctypes.byref(y_dim_2), ctypes.c_void_p(y.ctypes.data), ctypes.byref(xi_present), ctypes.byref(xi_dim_1), ctypes.byref(xi_dim_2), ctypes.c_void_p(xi.ctypes.data), ctypes.byref(info))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return info.value

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine EVALUATE
    
    def evaluate(self, config, model, x, y, xi=None, positions=None, embeddings=None):
        '''! Evaluate the piecewise linear regression model.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "model"
        if ((not issubclass(type(model), numpy.ndarray)) or
            (not numpy.asarray(model).flags.f_contiguous) or
            (not (model.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'model' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            model = numpy.asarray(model, dtype=ctypes.c_float, order='F')
        model_dim_1 = ctypes.c_int(model.shape[0])
        
        # Setting up "x"
        if ((not issubclass(type(x), numpy.ndarray)) or
            (not numpy.asarray(x).flags.f_contiguous) or
            (not (x.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'x' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            x = numpy.asarray(x, dtype=ctypes.c_float, order='F')
        x_dim_1 = ctypes.c_int(x.shape[0])
        x_dim_2 = ctypes.c_int(x.shape[1])
        
        # Setting up "xi"
        xi_present = ctypes.c_bool(True)
        if (xi is None):
            xi_present = ctypes.c_bool(False)
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif (type(xi) == bool) and (xi):
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif ((not issubclass(type(xi), numpy.ndarray)) or
              (not numpy.asarray(xi).flags.f_contiguous) or
              (not (xi.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'xi' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            xi = numpy.asarray(xi, dtype=ctypes.c_int, order='F')
        if (xi_present):
            xi_dim_1 = ctypes.c_int(xi.shape[0])
            xi_dim_2 = ctypes.c_int(xi.shape[1])
        else:
            xi_dim_1 = ctypes.c_int()
            xi_dim_2 = ctypes.c_int()
        
        # Setting up "y"
        if ((not issubclass(type(y), numpy.ndarray)) or
            (not numpy.asarray(y).flags.f_contiguous) or
            (not (y.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'y' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            y = numpy.asarray(y, dtype=ctypes.c_float, order='F')
        y_dim_1 = ctypes.c_int(y.shape[0])
        y_dim_2 = ctypes.c_int(y.shape[1])
        
        # Setting up "positions"
        positions_present = ctypes.c_bool(True)
        if (positions is None):
            positions_present = ctypes.c_bool(False)
            positions = numpy.zeros(shape=(1,1,1), dtype=ctypes.c_int, order='F')
        elif (type(positions) == bool) and (positions):
            positions = numpy.zeros(shape=(1,1,1), dtype=ctypes.c_int, order='F')
        elif ((not issubclass(type(positions), numpy.ndarray)) or
              (not numpy.asarray(positions).flags.f_contiguous) or
              (not (positions.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'positions' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            positions = numpy.asarray(positions, dtype=ctypes.c_int, order='F')
        if (positions_present):
            positions_dim_1 = ctypes.c_int(positions.shape[0])
            positions_dim_2 = ctypes.c_int(positions.shape[1])
            positions_dim_3 = ctypes.c_int(positions.shape[2])
        else:
            positions_dim_1 = ctypes.c_int()
            positions_dim_2 = ctypes.c_int()
            positions_dim_3 = ctypes.c_int()
        
        # Setting up "embeddings"
        embeddings_present = ctypes.c_bool(True)
        if (embeddings is None):
            embeddings_present = ctypes.c_bool(False)
            embeddings = numpy.zeros(shape=(1,1,1), dtype=ctypes.c_float, order='F')
        elif (type(embeddings) == bool) and (embeddings):
            embeddings = numpy.zeros(shape=(1,1,1), dtype=ctypes.c_float, order='F')
        elif ((not issubclass(type(embeddings), numpy.ndarray)) or
              (not numpy.asarray(embeddings).flags.f_contiguous) or
              (not (embeddings.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'embeddings' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            embeddings = numpy.asarray(embeddings, dtype=ctypes.c_float, order='F')
        if (embeddings_present):
            embeddings_dim_1 = ctypes.c_int(embeddings.shape[0])
            embeddings_dim_2 = ctypes.c_int(embeddings.shape[1])
            embeddings_dim_3 = ctypes.c_int(embeddings.shape[2])
        else:
            embeddings_dim_1 = ctypes.c_int()
            embeddings_dim_2 = ctypes.c_int()
            embeddings_dim_3 = ctypes.c_int()
        
        # Setting up "info"
        info = ctypes.c_int()
    
        # Call C-accessible Fortran wrapper.
        clib.c_evaluate(ctypes.byref(config), ctypes.byref(model_dim_1), ctypes.c_void_p(model.ctypes.data), ctypes.byref(x_dim_1), ctypes.byref(x_dim_2), ctypes.c_void_p(x.ctypes.data), ctypes.byref(xi_present), ctypes.byref(xi_dim_1), ctypes.byref(xi_dim_2), ctypes.c_void_p(xi.ctypes.data), ctypes.byref(y_dim_1), ctypes.byref(y_dim_2), ctypes.c_void_p(y.ctypes.data), ctypes.byref(positions_present), ctypes.byref(positions_dim_1), ctypes.byref(positions_dim_2), ctypes.byref(positions_dim_3), ctypes.c_void_p(positions.ctypes.data), ctypes.byref(embeddings_present), ctypes.byref(embeddings_dim_1), ctypes.byref(embeddings_dim_2), ctypes.byref(embeddings_dim_3), ctypes.c_void_p(embeddings.ctypes.data), ctypes.byref(info))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return y, (positions if positions_present else None), (embeddings if embeddings_present else None), info.value

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine MINIMIZE_MSE
    
    def minimize_mse(self, config, model, x, y, steps, xi=None, record=None):
        '''! Fit input / output pairs by minimizing mean squared error.'''
        MODEL_CONFIG = plrm.MODEL_CONFIG
        
        # Setting up "config"
        if (type(config) is not MODEL_CONFIG): config = MODEL_CONFIG(config)
        
        # Setting up "model"
        if ((not issubclass(type(model), numpy.ndarray)) or
            (not numpy.asarray(model).flags.f_contiguous) or
            (not (model.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'model' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            model = numpy.asarray(model, dtype=ctypes.c_float, order='F')
        model_dim_1 = ctypes.c_int(model.shape[0])
        
        # Setting up "x"
        if ((not issubclass(type(x), numpy.ndarray)) or
            (not numpy.asarray(x).flags.f_contiguous) or
            (not (x.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'x' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            x = numpy.asarray(x, dtype=ctypes.c_float, order='F')
        x_dim_1 = ctypes.c_int(x.shape[0])
        x_dim_2 = ctypes.c_int(x.shape[1])
        
        # Setting up "y"
        if ((not issubclass(type(y), numpy.ndarray)) or
            (not numpy.asarray(y).flags.f_contiguous) or
            (not (y.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'y' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            y = numpy.asarray(y, dtype=ctypes.c_float, order='F')
        y_dim_1 = ctypes.c_int(y.shape[0])
        y_dim_2 = ctypes.c_int(y.shape[1])
        
        # Setting up "steps"
        if (type(steps) is not ctypes.c_int): steps = ctypes.c_int(steps)
        
        # Setting up "xi"
        xi_present = ctypes.c_bool(True)
        if (xi is None):
            xi_present = ctypes.c_bool(False)
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif (type(xi) == bool) and (xi):
            xi = numpy.zeros(shape=(1,1), dtype=ctypes.c_int, order='F')
        elif ((not issubclass(type(xi), numpy.ndarray)) or
              (not numpy.asarray(xi).flags.f_contiguous) or
              (not (xi.dtype == numpy.dtype(ctypes.c_int)))):
            import warnings
            warnings.warn("The provided argument 'xi' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy).")
            xi = numpy.asarray(xi, dtype=ctypes.c_int, order='F')
        if (xi_present):
            xi_dim_1 = ctypes.c_int(xi.shape[0])
            xi_dim_2 = ctypes.c_int(xi.shape[1])
        else:
            xi_dim_1 = ctypes.c_int()
            xi_dim_2 = ctypes.c_int()
        
        # Setting up "sum_squared_error"
        sum_squared_error = ctypes.c_float()
        
        # Setting up "record"
        record_present = ctypes.c_bool(True)
        if (record is None):
            record_present = ctypes.c_bool(False)
            record = numpy.zeros(shape=(1,1), dtype=ctypes.c_float, order='F')
        elif (type(record) == bool) and (record):
            record = numpy.zeros(shape=(2, steps), dtype=ctypes.c_float, order='F')
        elif ((not issubclass(type(record), numpy.ndarray)) or
              (not numpy.asarray(record).flags.f_contiguous) or
              (not (record.dtype == numpy.dtype(ctypes.c_float)))):
            import warnings
            warnings.warn("The provided argument 'record' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
            record = numpy.asarray(record, dtype=ctypes.c_float, order='F')
        if (record_present):
            record_dim_1 = ctypes.c_int(record.shape[0])
            record_dim_2 = ctypes.c_int(record.shape[1])
        else:
            record_dim_1 = ctypes.c_int()
            record_dim_2 = ctypes.c_int()
        
        # Setting up "info"
        info = ctypes.c_int()
    
        # Call C-accessible Fortran wrapper.
        clib.c_minimize_mse(ctypes.byref(config), ctypes.byref(model_dim_1), ctypes.c_void_p(model.ctypes.data), ctypes.byref(x_dim_1), ctypes.byref(x_dim_2), ctypes.c_void_p(x.ctypes.data), ctypes.byref(y_dim_1), ctypes.byref(y_dim_2), ctypes.c_void_p(y.ctypes.data), ctypes.byref(steps), ctypes.byref(xi_present), ctypes.byref(xi_dim_1), ctypes.byref(xi_dim_2), ctypes.c_void_p(xi.ctypes.data), ctypes.byref(sum_squared_error), ctypes.byref(record_present), ctypes.byref(record_dim_1), ctypes.byref(record_dim_2), ctypes.c_void_p(record.ctypes.data), ctypes.byref(info))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return model, sum_squared_error.value, (record if record_present else None), info.value

plrm = plrm()

