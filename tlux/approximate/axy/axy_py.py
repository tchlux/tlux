import ctypes
import numpy as np
from tlux.approximate.axy.summary import AxyModel


# Cast the elements of an array to the specified data type.
#
# The function accepts an array of elements, a desired data type `dtype`, and an optional
# parameter `order` that specifies the memory layout of the resulting array. It attempts to
# cast each element of the array to the specified data type, preserving the shape of the input array.
# If `dtype` is passed as a type, then each element is cast to that type before creating the array.
# Otherwise, `dtype` should be a valid data type string recognizable by numpy.
#
# Parameters
# ----------
# arr : array_like
#     The input array whose elements are to be cast to the desired data type.
# dtype : type or str
#     The desired data type to which the elements of the input array are to be cast. 
#     Can be a type (e.g., int, float) or a string that numpy recognizes (e.g., 'int32', 'float64').
# order : {'C', 'F'}, optional, default: 'F'
#     Specify the memory layout of the output array. If 'C', the array will be in C-contiguous
#     order (row-major). If 'F', the array will be in Fortran-contiguous order (column-major).
#
# Returns
# -------
# ndarray
#     A numpy array with the same shape as the input array but with elements cast to the specified data type.
#
# Examples
# --------
# >>> cast([[1.2, 2.3], [3.4, 4.5]], int)
# array([[1, 2], [3, 4]])
# >>> cast([1, 2, 3], 'float64')
# array([1., 2., 3.])
# >>> cast([[1, 2], [3, 4]], complex, order='C')
# array([[1.+0.j, 2.+0.j], [3.+0.j, 4.+0.j]])
#
# Raises
# ------
# ValueError
#     If `dtype` is not recognized as a valid data type.
# TypeError
#     If `arr` is not array_like.
#
def cast(arr, dtype, order="F"):
    arr = np.asarray(arr)
    if (type(dtype) is type):
        arr = np.asarray(
            [dtype(v) for v in arr.flatten()],
            dtype=dtype,
            order=order
        ).reshape(arr.shape)
    else:
        arr = np.asarray(arr, dtype=dtype, order=order)
    return arr


# Convert a single index to a pair of indices.
# 
# Parameters
# ----------
# n : int
#     The number of source elements / the largest number in any pair.
# i : int
#     The single index to be converted to a pair of indices that
#     is 1-indexed and in the range [1, num_elements**2].
# 
# Returns
# -------
# tuple[int, int]
#   A pair of indices (pair1, pair2) that come from a function
#    the generates an ordering such that all pairs involving
#    smaller numbers come first.
# 
# Example
# -------
#  >>> [index_to_pair(3, i) for i in range(1,10)]
#  [(1,1) (1,2) (2,1) (1,3) (3,1) (2,2) (2,3) (3,2) (3,3)]
# 
def index_to_pair(num_elements, i):
    index_from_back = num_elements**2 - i
    group_from_back = int(index_from_back**(1/2))
    group = num_elements - group_from_back
    remaining_in_group = index_from_back - group_from_back**2
    group_size = 2*group_from_back + 1
    index_in_group = group_size - remaining_in_group
    other = group + index_in_group // 2
    if (index_in_group % 2 == 0):
        return (group, other)
    else:
        return (other, group)


# Check if the shapes and values of the input arrays match the expectations based on the model configuration.
# 
# Parameters
# ----------
# config : ModelConfig
#     Model configuration object with attributes:
#     - TOTAL_SIZE: expected total size of the model array.
#     - MDN: expected first dimension size of the X input array.
#     - MDO: expected first dimension size of the Y output array when greater than 0.
#     - ADO: expected first dimension size of the Y output array when MDO is 0.
#     - MNE: expected maximum value in the XI input array when greater than 0.
#     - ADI: flag that when greater than 0, SIZES array size must match Y second dimension size.
#     - PARTIAL_AGGREGATION: flag to indicate whether partial aggregation is allowed.
#     - ADN: expected first dimension size of the AX input array.
#     - ANE: expected maximum value in the AXI input array.
# 
# model : ndarray of shape (n,)
#     Model array of real numbers.
# 
# ax : ndarray of shape (adn, m)
#     AX input array of real numbers.
# 
# axi : ndarray of shape (p, m)
#     AXI input array of integers.
# 
# sizes : ndarray of shape (q,)
#     Sizes array of integers.
# 
# x : ndarray of shape (mdn, k)
#     X input array of real numbers.
# 
# xi : ndarray of shape (r, k)
#     XI input array of integers.
# 
# y : ndarray of shape (s, k)
#     Y input array of real numbers.
# 
# Returns
# -------
# info : int
#     Returns nonzero INFO if any shapes or values do not match expectations:
#     1: Model size does not match model configuration.
#     2: Input arrays do not match in size.
#     3: X input dimension is bad.
#     4: Model output dimension is bad, does not match Y.
#     5: Aggregator output dimension is bad, does not match Y.
#     6: Input integer XI size does not match X.
#     7: Input integer X index out of range.
#     8: SIZES has wrong size.
#     9: AX and SUM(SIZES) do not match.
#     10: AX input dimension is bad.
#     11: Input integer AXI size does not match AX.
#     12: Input integer AX index out of range.
# 
def check_shape(config, model, ax, axi, sizes, x, xi, y):
    # Compute whether the shape matches the CONFIG
    if (model.size != config.TOTAL_SIZE):
        return 1  # Model size does not match model configuration.
    if (x.shape[1] != y.shape[1]):
        return 2  # Input arrays do not match in size.
    if (x.shape[0] != config.MDN):
        return 3  # X input dimension is bad.
    if ((config.MDO > 0) and (y.shape[0] != config.MDO)):
        return 4  # Model output dimension is bad, does not match Y.
    if ((config.MDO == 0) and (y.shape[0] != config.ADO)):
        return 5  # Aggregator output dimension is bad, does not match Y.
    if ((config.MNE > 0) and (xi.shape[1] != x.shape[1])):
        return 6  # Input integer XI size does not match X.
    if ((xi.min() < 0) or (xi.max() > config.MNE)):
        return 7  # Input integer X index out of range.
    if ((config.ADI > 0) and (not config.PARTIAL_AGGREGATION) and (sizes.size != y.shape[1])):
        return 8  # SIZES has wrong size.
    if (ax.shape[1] != sizes.sum()):
        return 9  # AX and SUM(SIZES) do not match.
    if (ax.shape[0] != config.ADN):
        return 10  # AX input dimension is bad.
    if (axi.shape[1] != ax.shape[1]):
        return 11  # Input integer AXI size does not match AX.
    if ((axi.min() < 0) or (axi.max() > config.ANE)):
        return 12  # Input integer AX index out of range.
    return 0  # All checks passed.



# Normalize numeric input values.
# 
# Parameters
# ----------
# config : object
#     An object representing the model configuration, expected to have the following attributes:
#         NEEDS_SHIFTING : bool
#         NEEDS_SCALING : bool
#         ADN : int
#         MDN : int
#         AISS, AISE, MISS, MISE : int
#         AIMS, AIME, MIMS, MIME : int
#         ADO, MDO : int
#         NORMALIZE : bool
#         NEEDS_CLEANING : bool
# 
# model : ndarray
#     1D array of shape (n,) representing the model.
# 
# ax : ndarray
#     2D array of shape (m, p) representing aggregate inputs.
# 
# sizes : ndarray
#     1D array of shape (s,) representing sizes.
# 
# x : ndarray
#     2D array of shape (q, r) representing fixed inputs.
# 
# Returns
# -------
# ax : ndarray
#     2D array of normalized aggregate inputs.
# 
# x : ndarray
#     2D array of normalized fixed inputs.
# 
def normalize_inputs(config, model, ax, sizes, x):
    # If normalization needs to happen..
    if config.NORMALIZE:
        # Normalize the aggregate inputs.
        if (config.ADO > 0) and (config.ADN > 0):
            # Apply shift terms to aggregator inputs.
            if config.NEEDS_SHIFTING:
                ax_shift = model[config.AISS:config.AISE]
                ax[:config.ADN,:] += ax_shift[:, np.newaxis]

            # Remove any NaN or Inf values from the data.
            if config.NEEDS_CLEANING:
                ax[:config.ADN,:][np.isnan(ax[:config.ADN,:]) | ~np.isfinite(ax[:config.ADN,:])] = 0.0

            # Apply multiplier.
            if config.NEEDS_SCALING:
                ax_rescale = model[config.AIMS:config.AIME].reshape(config.ADN, config.ADN)
                ax[:config.ADN,:] = np.dot(ax_rescale.T, ax[:config.ADN,:])

        # Normalize the fixed inputs.
        if (config.MDO > 0) and (config.MDN > 0):
            # Apply shift terms to numeric model inputs.
            if config.NEEDS_SHIFTING:
                x_shift = model[config.MISS:config.MISE]
                x[:config.MDN,:] += x_shift[:, np.newaxis]

            # Remove any NaN or Inf values from the data.
            if config.NEEDS_CLEANING:
                x[:config.MDN,:][np.isnan(x[:config.MDN,:]) | ~np.isfinite(x[:config.MDN,:])] = 0.0

            # Apply multiplier.
            if config.NEEDS_SCALING:
                x_rescale = model[config.MIMS:config.MIME].reshape(config.MDN, config.MDN)
                x[:config.MDN,:] = np.dot(x_rescale.T, x[:config.MDN,:])

    return ax, x, 0




# Given the raw input data, fetch a new set of data that fits in memory.
# 
# Parameters
# ----------
# config : object
#     Config object, contains model configuration.
# agg_iterators_in : np.ndarray
#     2D numpy array of int64.
# ax_in : np.ndarray
#     2D numpy array of float.
# ax : np.ndarray
#     2D numpy array of float.
# axi_in : np.ndarray
#     2D numpy array of int64.
# axi : np.ndarray
#     2D numpy array of int64.
# sizes_in : np.ndarray
#     1D numpy array of int64.
# sizes : np.ndarray
#     1D numpy array of int64.
# x_in : np.ndarray
#     2D numpy array of float.
# x : np.ndarray
#     2D numpy array of float.
# xi_in : np.ndarray
#     2D numpy array of int64.
# xi : np.ndarray
#     2D numpy array of int64.
# y_in : np.ndarray
#     2D numpy array of float.
# y : np.ndarray
#     2D numpy array of float.
# yw_in : np.ndarray
#     2D numpy array of float.
# yw : np.ndarray
#     2D numpy array of float.
# 
# Returns
# -------
# na : int
#     Total number of aggregate inputs that were added.
# nm : int
#     Total number of inputs.
# 
def fetch_data(config, agg_iterators_in, ax_in, ax, axi_in, axi, sizes_in, sizes, x_in, x, xi_in, xi, y_in, y, yw_in, yw):
    # Local variables
    one = np.int64(1)
    zero = np.int64(0)
    rt = np.float64

    # Allocate storage space if it will be used.
    if sizes_in.size > 0:
        rsizes = np.empty(config.nm, dtype=rt)
        sorted_order = np.empty(config.nm, dtype=np.int64)

    # Pack the regular inputs into the working space, storing sizes.
    i_next = config.i_next
    i_mult = config.i_mult
    i_step = config.i_step
    i_iter = config.i_iter

    for i in range(1, min(config.nm, y_in.shape[1]) + 1):
        # Choose iteration strategy
        if config.nm >= y_in.shape[1] and not config.partial_aggregation:
            gendex = i
        else:
            gendex = get_next_index(config.nmt, config.i_next, config.i_mult,
                                    config.i_step, config.i_mod, config.i_iter)
        # Store the size
        if sizes_in.size > 0:
            rsizes[i - 1] = rt(sizes_in[gendex - 1])
            if config.pairwise_aggregation:
                rsizes[i - 1] = rsizes[i - 1]**2

        x[:config.mdn, i - 1] = x_in[:, gendex - 1]
        xi[:, i - 1] = xi_in[:, gendex - 1]
        y[:, i - 1] = y_in[:, gendex - 1]
        yw[:, i - 1] = yw_in[:, gendex - 1]

    # TODO: Get the aggregate inputs too.



# Define the full EVALUATE function in python (reimplementation).
def evaluate(config, model, ax, axi, sizes, x, xi, dtype="float32", **unused_kwargs):
    # Get some constants.
    nm = x.shape[1]
    na = ax.shape[1]
    m = AxyModel(config, cast(model, dtype))
    state_values = {}
    # Embed the AXI values.
    ax_embedded = cast(np.zeros((config.adi, na)), dtype)
    for n in range(axi.shape[1]):
        ax_embedded[:config.adn,n] = ax[:config.adn,n]
        for d in range(axi.shape[0]):
            e = axi[d,n]
            if (e > 0) and (e <= config.ane):
                ax_embedded[-config.ade:,n] += m.a_embeddings[:,e-1]
            elif (e > config.ane):
                e1, e2 = index_to_pair(num_elements=config.ane+1, i=e-config.ane)
                ax_embedded[-config.ade:,n] += m.a_embeddings[:,e1-1-1] - m.a_embeddings[:,e2-1-1]
        if (axi.shape[0] > 1):
            ax_embedded[-config.ade:,n] /= axi.shape[0]
    ax = ax_embedded
    state_values["ax"] = ax
    # Embed the XI values.
    x_embedded = cast(np.zeros((config.mdi, nm)), dtype)
    for n in range(xi.shape[1]):
        x_embedded[:config.mdn,n] = x[:config.mdn,n]
        for d in range(xi.shape[0]):
            e = xi[d,n]
            if (e > 0) and (e <= config.mne):
                x_embedded[config.mdn:config.mdn+config.mde:,n] += m.m_embeddings[:,e-1]
            elif (e > config.mne):
                e1, e2 = index_to_pair(num_elements=config.mne, i=e)
                x_embedded[config.mdn:config.mdn+config.mde,n] += m.m_embeddings[:,e1-1] - m.m_embeddings[:,e2-1]
        if (xi.shape[0] > 1):
            x_embedded[-config.mde:,n] /= xi.shape[0]
    x = x_embedded
    state_values["x"] = x
    # Initialize a holder for the output.
    y = cast(np.zeros((config.do, nm)), dtype)
    # Evaluate the aggregator.
    if (config.ado > 0):
        if (config.normalize and (config.adn > 0)):
            # Add the shift term (across all point column vectors).
            ax[:config.adn,:] = (ax[:config.adn,:].T + m.ax_shift).T
            # Replace all NaN or Inf values
            ax[:config.adn,:] = np.where(
                np.logical_or(
                    np.isnan(ax[:config.adn,:].astype("float32")),
                    np.isinf(ax[:config.adn,:].astype("float32")),
                ),
                0.0,
                ax[:config.adn,:]
            )
            # Apply the input multiplier.
            if (config.needs_scaling):
                ax[:config.adn,:] = m.ax_rescale.T @ ax[:config.adn,:]
        # Evaluate the MLP.
        values = ax
        state_values["a_states"] = []
        if (config.ans > 0):
            # Apply input transformation.
            values = np.clip(
                ((values.T @ m.a_input_vecs) + m.a_input_shift).T,
                config.discontinuity, float('inf')
            )
            state_values["a_states"].append(values)
            # Apply internal transformations.
            for i in range(config.ans-1):
                values = np.clip(
                    ((values.T @ m.a_state_vecs[:,:,i]) + m.a_state_shift[:,i]).T,
                    config.discontinuity, float('inf')
                )
                state_values["a_states"].append(values)
        state_values["a_states"] = np.asarray(state_values["a_states"]).T
        # Apply output transformation.
        ay = (values.T @ m.a_output_vecs[:,:])
        state_values["ay"] = ay
        ay_error = ay[:,config.ado:config.ado+1]  # extract +1 for error prediction
        ay = ay[:,:config.ado] # strip off +1 for error prediction
        # If there is a following model..
        if (config.mdo > 0):
            # Apply output shift.
            ay[:,:] = (ay + m.ay_shift)
            # Compute the first aggregator output embedding position.
            e = config.mdn + config.mde
            # Set the aggregator output to be a slice of X.
            agg_out = x[e:,:]
        else:
            # Set the aggregator output to be Y.
            agg_out = y[:,:]
        # Aggregate the batches. With partial aggregation, we have one output for
        #  partial mean starting from the "last" aggregate output.
        if (config.partial_aggregation):
            f_start = 0
            a_start = 0
            for i,s in enumerate(sizes):
                a_end = a_start + s
                f_end = f_start + max(1,s)
                for out_i, agg_i in zip(range(f_end-1, f_start-1, -1), range(a_end-1, a_start-1, -1)):
                    num_elements = a_end - agg_i
                    agg_out[:,out_i] = ay[agg_i:a_end,:config.ado].sum(axis=0) / num_elements
                # When there is no size, a zero value is assigned.
                if (s == 0):
                    agg_out[:,f_end-1] = 0.0
                # Transition the start for the next element.
                a_start = a_end
                f_start = f_end
        # Without partial aggregation, we only compute the mean all outputs in each group.
        else:
            a = 0
            for i,s in enumerate(sizes):
                if (s > 0):
                    agg_out[:,i] = ay[a:a+s,:config.ado].sum(axis=0) / s
                else:
                    agg_out[:,i] = 0
                a += s
    # Evaluate the fixed model.
    if (config.mdo > 0):
        if (config.normalize and (config.mdn > 0)):
            # Add the shift term (across all point column vectors).
            x[:config.mdn,:] = (x[:config.mdn,:].T + m.x_shift).T
            # Replace all NaN or Inf values
            x[:config.mdn,:] = np.where(
                np.logical_or(
                    np.isnan(x[:config.mdn,:].astype("float32")),
                    np.isinf(x[:config.mdn,:].astype("float32")),
                ),
                0.0,
                x[:config.mdn,:]
            )
            # Apply the input multiplier.
            if (config.needs_scaling):
                x[:config.mdn,:] = m.x_rescale.T @ x[:config.mdn,:]
        # Evaluate the MLP.
        values = x
        state_values["m_states"] = []
        if (config.mns > 0):
            # Apply input transformation.
            values = np.clip(
                ((values.T @ m.m_input_vecs) + m.m_input_shift).T,
                config.discontinuity, float('inf')
            )
            state_values["m_states"].append(values)
            # Apply internal transformations.
            for i in range(config.mns-1):
                values = np.clip(
                    ((values.T @ m.m_state_vecs[:,:,i]) + m.m_state_shift[:,i]).T,
                    config.discontinuity, float('inf')
                )
                state_values["m_states"].append(values)
        state_values["m_states"] = np.asarray(state_values["m_states"]).T
        # Apply output transformation.
        y = (values.T @ m.m_output_vecs[:,:]).T
    # Apply final normalization.
    if (config.normalize):
        if (config.needs_scaling):
            y[:config.do-config.doe,:] = m.y_rescale.T @ y[:config.do-config.doe,:]
        y[:config.do-config.doe,:] = (y[:config.do-config.doe,:].T + m.y_shift).T
    state_values["y"] = y
    # Return the final values.
    return state_values


# This defines a C structure that can be used to hold this defined type.
class MODEL_CONFIG(ctypes.Structure):
    # (name, ctype) fields for this structure.
    _fields_ = [("adn", ctypes.c_int), ("ade", ctypes.c_int), ("ane", ctypes.c_int), ("ads", ctypes.c_int), ("ans", ctypes.c_int), ("ado", ctypes.c_int), ("adi", ctypes.c_int), ("adso", ctypes.c_int), ("mdn", ctypes.c_int), ("mde", ctypes.c_int), ("mne", ctypes.c_int), ("mds", ctypes.c_int), ("mns", ctypes.c_int), ("mdo", ctypes.c_int), ("mdi", ctypes.c_int), ("mdso", ctypes.c_int), ("do", ctypes.c_int), ("total_size", ctypes.c_long), ("num_vars", ctypes.c_long), ("asev", ctypes.c_long), ("aeev", ctypes.c_long), ("asiv", ctypes.c_long), ("aeiv", ctypes.c_long), ("asis", ctypes.c_long), ("aeis", ctypes.c_long), ("assv", ctypes.c_long), ("aesv", ctypes.c_long), ("asss", ctypes.c_long), ("aess", ctypes.c_long), ("asov", ctypes.c_long), ("aeov", ctypes.c_long), ("msev", ctypes.c_long), ("meev", ctypes.c_long), ("msiv", ctypes.c_long), ("meiv", ctypes.c_long), ("msis", ctypes.c_long), ("meis", ctypes.c_long), ("mssv", ctypes.c_long), ("mesv", ctypes.c_long), ("msss", ctypes.c_long), ("mess", ctypes.c_long), ("msov", ctypes.c_long), ("meov", ctypes.c_long), ("aiss", ctypes.c_long), ("aise", ctypes.c_long), ("aoss", ctypes.c_long), ("aose", ctypes.c_long), ("aims", ctypes.c_long), ("aime", ctypes.c_long), ("aecs", ctypes.c_long), ("aece", ctypes.c_long), ("miss", ctypes.c_long), ("mise", ctypes.c_long), ("moss", ctypes.c_long), ("mose", ctypes.c_long), ("mims", ctypes.c_long), ("mime", ctypes.c_long), ("moms", ctypes.c_long), ("mome", ctypes.c_long), ("mecs", ctypes.c_long), ("mece", ctypes.c_long), ("discontinuity", ctypes.c_float), ("max_step_factor", ctypes.c_float), ("step_factor", ctypes.c_float), ("min_step_factor", ctypes.c_float), ("max_step_component", ctypes.c_float), ("max_curv_component", ctypes.c_float), ("min_curv_component", ctypes.c_float), ("faster_rate", ctypes.c_float), ("slower_rate", ctypes.c_float), ("min_update_ratio", ctypes.c_float), ("update_ratio_step", ctypes.c_float), ("step_mean_change", ctypes.c_float), ("step_curv_change", ctypes.c_float), ("step_ay_change", ctypes.c_float), ("step_emb_change", ctypes.c_float), ("initial_curv_estimate", ctypes.c_float), ("mse_upper_limit", ctypes.c_float), ("min_steps_to_stability", ctypes.c_long), ("max_batch", ctypes.c_long), ("num_threads", ctypes.c_long), ("print_delay_sec", ctypes.c_long), ("steps_taken", ctypes.c_long), ("condition_frequency", ctypes.c_long), ("log_grad_norm_frequency", ctypes.c_long), ("rank_check_frequency", ctypes.c_long), ("num_to_update", ctypes.c_long), ("keep_best", ctypes.c_bool), ("early_stop", ctypes.c_bool), ("basis_replacement", ctypes.c_bool), ("reshuffle", ctypes.c_bool), ("granular_parallelism", ctypes.c_bool), ("partial_aggregation", ctypes.c_bool), ("pairwise_aggregation", ctypes.c_bool), ("ax_normalized", ctypes.c_bool), ("rescale_ax", ctypes.c_bool), ("axi_normalized", ctypes.c_bool), ("ay_normalized", ctypes.c_bool), ("x_normalized", ctypes.c_bool), ("rescale_x", ctypes.c_bool), ("xi_normalized", ctypes.c_bool), ("y_normalized", ctypes.c_bool), ("rescale_y", ctypes.c_bool), ("encode_scaling", ctypes.c_bool), ("normalize", ctypes.c_bool), ("needs_shifting", ctypes.c_bool), ("needs_cleaning", ctypes.c_bool), ("needs_scaling", ctypes.c_bool), ("rwork_size", ctypes.c_long), ("iwork_size", ctypes.c_long), ("lwork_size", ctypes.c_long), ("na", ctypes.c_long), ("nat", ctypes.c_long), ("nm", ctypes.c_long), ("nms", ctypes.c_long), ("nmt", ctypes.c_long), ("i_next", ctypes.c_long), ("i_step", ctypes.c_long), ("i_mult", ctypes.c_long), ("i_mod", ctypes.c_long), ("i_iter", ctypes.c_long), ("smg", ctypes.c_long), ("emg", ctypes.c_long), ("smgm", ctypes.c_long), ("emgm", ctypes.c_long), ("smgc", ctypes.c_long), ("emgc", ctypes.c_long), ("sbm", ctypes.c_long), ("ebm", ctypes.c_long), ("saxb", ctypes.c_long), ("eaxb", ctypes.c_long), ("say", ctypes.c_long), ("eay", ctypes.c_long), ("smxb", ctypes.c_long), ("emxb", ctypes.c_long), ("smyb", ctypes.c_long), ("emyb", ctypes.c_long), ("saet", ctypes.c_long), ("eaet", ctypes.c_long), ("saxs", ctypes.c_long), ("eaxs", ctypes.c_long), ("saxg", ctypes.c_long), ("eaxg", ctypes.c_long), ("sayg", ctypes.c_long), ("eayg", ctypes.c_long), ("smet", ctypes.c_long), ("emet", ctypes.c_long), ("smxs", ctypes.c_long), ("emxs", ctypes.c_long), ("smxg", ctypes.c_long), ("emxg", ctypes.c_long), ("syg", ctypes.c_long), ("eyg", ctypes.c_long), ("saxis", ctypes.c_long), ("eaxis", ctypes.c_long), ("saxir", ctypes.c_long), ("eaxir", ctypes.c_long), ("smxis", ctypes.c_long), ("emxis", ctypes.c_long), ("smxir", ctypes.c_long), ("emxir", ctypes.c_long), ("sal", ctypes.c_long), ("eal", ctypes.c_long), ("sml", ctypes.c_long), ("eml", ctypes.c_long), ("sast", ctypes.c_long), ("east", ctypes.c_long), ("smst", ctypes.c_long), ("emst", ctypes.c_long), ("saxi", ctypes.c_long), ("eaxi", ctypes.c_long), ("smxi", ctypes.c_long), ("emxi", ctypes.c_long), ("ssb", ctypes.c_long), ("esb", ctypes.c_long), ("sao", ctypes.c_long), ("eao", ctypes.c_long), ("smo", ctypes.c_long), ("emo", ctypes.c_long), ("sui", ctypes.c_long), ("eui", ctypes.c_long), ("wint", ctypes.c_long), ("cint", ctypes.c_long), ("wfit", ctypes.c_long), ("cfit", ctypes.c_long), ("wnrm", ctypes.c_long), ("cnrm", ctypes.c_long), ("wgen", ctypes.c_long), ("cgen", ctypes.c_long), ("wemb", ctypes.c_long), ("cemb", ctypes.c_long), ("wevl", ctypes.c_long), ("cevl", ctypes.c_long), ("wgrd", ctypes.c_long), ("cgrd", ctypes.c_long), ("wrat", ctypes.c_long), ("crat", ctypes.c_long), ("wopt", ctypes.c_long), ("copt", ctypes.c_long), ("wcon", ctypes.c_long), ("ccon", ctypes.c_long), ("wrec", ctypes.c_long), ("crec", ctypes.c_long), ("wenc", ctypes.c_long), ("cenc", ctypes.c_long)]
    # Define an "__init__" that can take a class or keyword arguments as input.
    def __init__(self, value=0, **kwargs):
        # From whatever object (or dictionary) was given, assign internal values.
        self.adn = kwargs.get("adn", getattr(value, "adn", value))
        self.ade = kwargs.get("ade", getattr(value, "ade", value))
        self.ane = kwargs.get("ane", getattr(value, "ane", value))
        self.ads = kwargs.get("ads", getattr(value, "ads", value))
        self.ans = kwargs.get("ans", getattr(value, "ans", value))
        self.ado = kwargs.get("ado", getattr(value, "ado", value))
        self.adi = kwargs.get("adi", getattr(value, "adi", value))
        self.adso = kwargs.get("adso", getattr(value, "adso", value))
        self.mdn = kwargs.get("mdn", getattr(value, "mdn", value))
        self.mde = kwargs.get("mde", getattr(value, "mde", value))
        self.mne = kwargs.get("mne", getattr(value, "mne", value))
        self.mds = kwargs.get("mds", getattr(value, "mds", value))
        self.mns = kwargs.get("mns", getattr(value, "mns", value))
        self.mdo = kwargs.get("mdo", getattr(value, "mdo", value))
        self.mdi = kwargs.get("mdi", getattr(value, "mdi", value))
        self.mdso = kwargs.get("mdso", getattr(value, "mdso", value))
        self.do = kwargs.get("do", getattr(value, "do", value))
        self.total_size = kwargs.get("total_size", getattr(value, "total_size", value))
        self.num_vars = kwargs.get("num_vars", getattr(value, "num_vars", value))
        self.asev = kwargs.get("asev", getattr(value, "asev", value))
        self.aeev = kwargs.get("aeev", getattr(value, "aeev", value))
        self.asiv = kwargs.get("asiv", getattr(value, "asiv", value))
        self.aeiv = kwargs.get("aeiv", getattr(value, "aeiv", value))
        self.asis = kwargs.get("asis", getattr(value, "asis", value))
        self.aeis = kwargs.get("aeis", getattr(value, "aeis", value))
        self.assv = kwargs.get("assv", getattr(value, "assv", value))
        self.aesv = kwargs.get("aesv", getattr(value, "aesv", value))
        self.asss = kwargs.get("asss", getattr(value, "asss", value))
        self.aess = kwargs.get("aess", getattr(value, "aess", value))
        self.asov = kwargs.get("asov", getattr(value, "asov", value))
        self.aeov = kwargs.get("aeov", getattr(value, "aeov", value))
        self.msev = kwargs.get("msev", getattr(value, "msev", value))
        self.meev = kwargs.get("meev", getattr(value, "meev", value))
        self.msiv = kwargs.get("msiv", getattr(value, "msiv", value))
        self.meiv = kwargs.get("meiv", getattr(value, "meiv", value))
        self.msis = kwargs.get("msis", getattr(value, "msis", value))
        self.meis = kwargs.get("meis", getattr(value, "meis", value))
        self.mssv = kwargs.get("mssv", getattr(value, "mssv", value))
        self.mesv = kwargs.get("mesv", getattr(value, "mesv", value))
        self.msss = kwargs.get("msss", getattr(value, "msss", value))
        self.mess = kwargs.get("mess", getattr(value, "mess", value))
        self.msov = kwargs.get("msov", getattr(value, "msov", value))
        self.meov = kwargs.get("meov", getattr(value, "meov", value))
        self.aiss = kwargs.get("aiss", getattr(value, "aiss", value))
        self.aise = kwargs.get("aise", getattr(value, "aise", value))
        self.aoss = kwargs.get("aoss", getattr(value, "aoss", value))
        self.aose = kwargs.get("aose", getattr(value, "aose", value))
        self.aims = kwargs.get("aims", getattr(value, "aims", value))
        self.aime = kwargs.get("aime", getattr(value, "aime", value))
        self.aecs = kwargs.get("aecs", getattr(value, "aecs", value))
        self.aece = kwargs.get("aece", getattr(value, "aece", value))
        self.miss = kwargs.get("miss", getattr(value, "miss", value))
        self.mise = kwargs.get("mise", getattr(value, "mise", value))
        self.moss = kwargs.get("moss", getattr(value, "moss", value))
        self.mose = kwargs.get("mose", getattr(value, "mose", value))
        self.mims = kwargs.get("mims", getattr(value, "mims", value))
        self.mime = kwargs.get("mime", getattr(value, "mime", value))
        self.moms = kwargs.get("moms", getattr(value, "moms", value))
        self.mome = kwargs.get("mome", getattr(value, "mome", value))
        self.mecs = kwargs.get("mecs", getattr(value, "mecs", value))
        self.mece = kwargs.get("mece", getattr(value, "mece", value))
        self.discontinuity = kwargs.get("discontinuity", getattr(value, "discontinuity", value))
        self.max_step_factor = kwargs.get("max_step_factor", getattr(value, "max_step_factor", value))
        self.step_factor = kwargs.get("step_factor", getattr(value, "step_factor", value))
        self.min_step_factor = kwargs.get("min_step_factor", getattr(value, "min_step_factor", value))
        self.max_step_component = kwargs.get("max_step_component", getattr(value, "max_step_component", value))
        self.max_curv_component = kwargs.get("max_curv_component", getattr(value, "max_curv_component", value))
        self.min_curv_component = kwargs.get("min_curv_component", getattr(value, "min_curv_component", value))
        self.faster_rate = kwargs.get("faster_rate", getattr(value, "faster_rate", value))
        self.slower_rate = kwargs.get("slower_rate", getattr(value, "slower_rate", value))
        self.min_update_ratio = kwargs.get("min_update_ratio", getattr(value, "min_update_ratio", value))
        self.update_ratio_step = kwargs.get("update_ratio_step", getattr(value, "update_ratio_step", value))
        self.step_mean_change = kwargs.get("step_mean_change", getattr(value, "step_mean_change", value))
        self.step_curv_change = kwargs.get("step_curv_change", getattr(value, "step_curv_change", value))
        self.step_ay_change = kwargs.get("step_ay_change", getattr(value, "step_ay_change", value))
        self.step_emb_change = kwargs.get("step_emb_change", getattr(value, "step_emb_change", value))
        self.initial_curv_estimate = kwargs.get("initial_curv_estimate", getattr(value, "initial_curv_estimate", value))
        self.mse_upper_limit = kwargs.get("mse_upper_limit", getattr(value, "mse_upper_limit", value))
        self.min_steps_to_stability = kwargs.get("min_steps_to_stability", getattr(value, "min_steps_to_stability", value))
        self.max_batch = kwargs.get("max_batch", getattr(value, "max_batch", value))
        self.num_threads = kwargs.get("num_threads", getattr(value, "num_threads", value))
        self.print_delay_sec = kwargs.get("print_delay_sec", getattr(value, "print_delay_sec", value))
        self.steps_taken = kwargs.get("steps_taken", getattr(value, "steps_taken", value))
        self.condition_frequency = kwargs.get("condition_frequency", getattr(value, "condition_frequency", value))
        self.log_grad_norm_frequency = kwargs.get("log_grad_norm_frequency", getattr(value, "log_grad_norm_frequency", value))
        self.rank_check_frequency = kwargs.get("rank_check_frequency", getattr(value, "rank_check_frequency", value))
        self.num_to_update = kwargs.get("num_to_update", getattr(value, "num_to_update", value))
        self.keep_best = kwargs.get("keep_best", getattr(value, "keep_best", value))
        self.early_stop = kwargs.get("early_stop", getattr(value, "early_stop", value))
        self.basis_replacement = kwargs.get("basis_replacement", getattr(value, "basis_replacement", value))
        self.reshuffle = kwargs.get("reshuffle", getattr(value, "reshuffle", value))
        self.granular_parallelism = kwargs.get("granular_parallelism", getattr(value, "granular_parallelism", value))
        self.partial_aggregation = kwargs.get("partial_aggregation", getattr(value, "partial_aggregation", value))
        self.pairwise_aggregation = kwargs.get("pairwise_aggregation", getattr(value, "pairwise_aggregation", value))
        self.ax_normalized = kwargs.get("ax_normalized", getattr(value, "ax_normalized", value))
        self.rescale_ax = kwargs.get("rescale_ax", getattr(value, "rescale_ax", value))
        self.axi_normalized = kwargs.get("axi_normalized", getattr(value, "axi_normalized", value))
        self.ay_normalized = kwargs.get("ay_normalized", getattr(value, "ay_normalized", value))
        self.x_normalized = kwargs.get("x_normalized", getattr(value, "x_normalized", value))
        self.rescale_x = kwargs.get("rescale_x", getattr(value, "rescale_x", value))
        self.xi_normalized = kwargs.get("xi_normalized", getattr(value, "xi_normalized", value))
        self.y_normalized = kwargs.get("y_normalized", getattr(value, "y_normalized", value))
        self.rescale_y = kwargs.get("rescale_y", getattr(value, "rescale_y", value))
        self.encode_scaling = kwargs.get("encode_scaling", getattr(value, "encode_scaling", value))
        self.normalize = kwargs.get("normalize", getattr(value, "normalize", value))
        self.needs_shifting = kwargs.get("needs_shifting", getattr(value, "needs_shifting", value))
        self.needs_cleaning = kwargs.get("needs_cleaning", getattr(value, "needs_cleaning", value))
        self.needs_scaling = kwargs.get("needs_scaling", getattr(value, "needs_scaling", value))
        self.rwork_size = kwargs.get("rwork_size", getattr(value, "rwork_size", value))
        self.iwork_size = kwargs.get("iwork_size", getattr(value, "iwork_size", value))
        self.lwork_size = kwargs.get("lwork_size", getattr(value, "lwork_size", value))
        self.na = kwargs.get("na", getattr(value, "na", value))
        self.nat = kwargs.get("nat", getattr(value, "nat", value))
        self.nm = kwargs.get("nm", getattr(value, "nm", value))
        self.nms = kwargs.get("nms", getattr(value, "nms", value))
        self.nmt = kwargs.get("nmt", getattr(value, "nmt", value))
        self.i_next = kwargs.get("i_next", getattr(value, "i_next", value))
        self.i_step = kwargs.get("i_step", getattr(value, "i_step", value))
        self.i_mult = kwargs.get("i_mult", getattr(value, "i_mult", value))
        self.i_mod = kwargs.get("i_mod", getattr(value, "i_mod", value))
        self.i_iter = kwargs.get("i_iter", getattr(value, "i_iter", value))
        self.smg = kwargs.get("smg", getattr(value, "smg", value))
        self.emg = kwargs.get("emg", getattr(value, "emg", value))
        self.smgm = kwargs.get("smgm", getattr(value, "smgm", value))
        self.emgm = kwargs.get("emgm", getattr(value, "emgm", value))
        self.smgc = kwargs.get("smgc", getattr(value, "smgc", value))
        self.emgc = kwargs.get("emgc", getattr(value, "emgc", value))
        self.sbm = kwargs.get("sbm", getattr(value, "sbm", value))
        self.ebm = kwargs.get("ebm", getattr(value, "ebm", value))
        self.saxb = kwargs.get("saxb", getattr(value, "saxb", value))
        self.eaxb = kwargs.get("eaxb", getattr(value, "eaxb", value))
        self.say = kwargs.get("say", getattr(value, "say", value))
        self.eay = kwargs.get("eay", getattr(value, "eay", value))
        self.smxb = kwargs.get("smxb", getattr(value, "smxb", value))
        self.emxb = kwargs.get("emxb", getattr(value, "emxb", value))
        self.smyb = kwargs.get("smyb", getattr(value, "smyb", value))
        self.emyb = kwargs.get("emyb", getattr(value, "emyb", value))
        self.saet = kwargs.get("saet", getattr(value, "saet", value))
        self.eaet = kwargs.get("eaet", getattr(value, "eaet", value))
        self.saxs = kwargs.get("saxs", getattr(value, "saxs", value))
        self.eaxs = kwargs.get("eaxs", getattr(value, "eaxs", value))
        self.saxg = kwargs.get("saxg", getattr(value, "saxg", value))
        self.eaxg = kwargs.get("eaxg", getattr(value, "eaxg", value))
        self.sayg = kwargs.get("sayg", getattr(value, "sayg", value))
        self.eayg = kwargs.get("eayg", getattr(value, "eayg", value))
        self.smet = kwargs.get("smet", getattr(value, "smet", value))
        self.emet = kwargs.get("emet", getattr(value, "emet", value))
        self.smxs = kwargs.get("smxs", getattr(value, "smxs", value))
        self.emxs = kwargs.get("emxs", getattr(value, "emxs", value))
        self.smxg = kwargs.get("smxg", getattr(value, "smxg", value))
        self.emxg = kwargs.get("emxg", getattr(value, "emxg", value))
        self.syg = kwargs.get("syg", getattr(value, "syg", value))
        self.eyg = kwargs.get("eyg", getattr(value, "eyg", value))
        self.saxis = kwargs.get("saxis", getattr(value, "saxis", value))
        self.eaxis = kwargs.get("eaxis", getattr(value, "eaxis", value))
        self.saxir = kwargs.get("saxir", getattr(value, "saxir", value))
        self.eaxir = kwargs.get("eaxir", getattr(value, "eaxir", value))
        self.smxis = kwargs.get("smxis", getattr(value, "smxis", value))
        self.emxis = kwargs.get("emxis", getattr(value, "emxis", value))
        self.smxir = kwargs.get("smxir", getattr(value, "smxir", value))
        self.emxir = kwargs.get("emxir", getattr(value, "emxir", value))
        self.sal = kwargs.get("sal", getattr(value, "sal", value))
        self.eal = kwargs.get("eal", getattr(value, "eal", value))
        self.sml = kwargs.get("sml", getattr(value, "sml", value))
        self.eml = kwargs.get("eml", getattr(value, "eml", value))
        self.sast = kwargs.get("sast", getattr(value, "sast", value))
        self.east = kwargs.get("east", getattr(value, "east", value))
        self.smst = kwargs.get("smst", getattr(value, "smst", value))
        self.emst = kwargs.get("emst", getattr(value, "emst", value))
        self.saxi = kwargs.get("saxi", getattr(value, "saxi", value))
        self.eaxi = kwargs.get("eaxi", getattr(value, "eaxi", value))
        self.smxi = kwargs.get("smxi", getattr(value, "smxi", value))
        self.emxi = kwargs.get("emxi", getattr(value, "emxi", value))
        self.ssb = kwargs.get("ssb", getattr(value, "ssb", value))
        self.esb = kwargs.get("esb", getattr(value, "esb", value))
        self.sao = kwargs.get("sao", getattr(value, "sao", value))
        self.eao = kwargs.get("eao", getattr(value, "eao", value))
        self.smo = kwargs.get("smo", getattr(value, "smo", value))
        self.emo = kwargs.get("emo", getattr(value, "emo", value))
        self.sui = kwargs.get("sui", getattr(value, "sui", value))
        self.eui = kwargs.get("eui", getattr(value, "eui", value))
        self.wint = kwargs.get("wint", getattr(value, "wint", value))
        self.cint = kwargs.get("cint", getattr(value, "cint", value))
        self.wfit = kwargs.get("wfit", getattr(value, "wfit", value))
        self.cfit = kwargs.get("cfit", getattr(value, "cfit", value))
        self.wnrm = kwargs.get("wnrm", getattr(value, "wnrm", value))
        self.cnrm = kwargs.get("cnrm", getattr(value, "cnrm", value))
        self.wgen = kwargs.get("wgen", getattr(value, "wgen", value))
        self.cgen = kwargs.get("cgen", getattr(value, "cgen", value))
        self.wemb = kwargs.get("wemb", getattr(value, "wemb", value))
        self.cemb = kwargs.get("cemb", getattr(value, "cemb", value))
        self.wevl = kwargs.get("wevl", getattr(value, "wevl", value))
        self.cevl = kwargs.get("cevl", getattr(value, "cevl", value))
        self.wgrd = kwargs.get("wgrd", getattr(value, "wgrd", value))
        self.cgrd = kwargs.get("cgrd", getattr(value, "cgrd", value))
        self.wrat = kwargs.get("wrat", getattr(value, "wrat", value))
        self.crat = kwargs.get("crat", getattr(value, "crat", value))
        self.wopt = kwargs.get("wopt", getattr(value, "wopt", value))
        self.copt = kwargs.get("copt", getattr(value, "copt", value))
        self.wcon = kwargs.get("wcon", getattr(value, "wcon", value))
        self.ccon = kwargs.get("ccon", getattr(value, "ccon", value))
        self.wrec = kwargs.get("wrec", getattr(value, "wrec", value))
        self.crec = kwargs.get("crec", getattr(value, "crec", value))
        self.wenc = kwargs.get("wenc", getattr(value, "wenc", value))
        self.cenc = kwargs.get("cenc", getattr(value, "cenc", value))
    # Define a "__str__" that produces a legible summary of this type.
    def __str__(self):
        s = []
        for (n, t) in self._fields_:
            s.append( n + "=" + str(getattr(self,n)) )
        return "MODEL_CONFIG[" + ", ".join(s) + "]"
    # Define an "__eq__" method that checks equality of all fields.
    def __eq__(self, other):
        for (n, t) in self._fields_:
            if (getattr(self, n) != getattr(other, n, None)):
                return False
        return True
