from tlux.unique import unique, to_int
import multiprocessing
import numpy as np


# Convert a numpy array to a set of unique values.
def to_set(array): return set(array.tolist())


# Insert 'value' into 'array' in sorted order, left of anything
#  with equal 'key' if 'unique=False', otherwise skipped for equal keys.
def insert_sorted(array, value, key=lambda i:i, unique=True):
    low = 0
    high = len(array)
    index = (low + high) // 2
    while high != low:
        if key(array[index]) > key(value):   high = index
        elif key(array[index]) < key(value):  low = index + 1
        elif unique: return
        index = (low + high) // 2
    if ((len(array) == 0) or (array[index] != value) or (not unique)):
        array.insert(index, value)


# Given a categorical input array, construct a dictionary for
#  mapping the unique values in the columns of the array to integers.
def i_map(xi, xi_map):
    # Generate the map (ordered list of unique values).
    if (len(xi.dtype) > 0):
        for n in xi.dtype.names:
            xi_map.append(xi[n]) 
    else:
        for i in range(xi.shape[1]):
            xi_map.append(xi[:,i])
    # Use parallel processing to get the unique values from the set.
    for i in range(len(xi_map)):
        xi_map[i] = unique(xi_map[i])
    # Return the map and the lookup.
    return xi_map


# Given a categorical input array (either 2D or struct), map this
#  array to an integer encoding matrix with the same number of
#  columns, but unique integers assigned to each unique value.
def i_encode(xi, xi_map):
    xi_rows = xi.shape[0]
    xi_cols = len(xi.dtype) or xi.shape[1]
    _xi = np.zeros((xi_rows, xi_cols), dtype="int64", order="C")
    base = 0
    for i, i_map in enumerate(xi_map):
        # Assign all the integer embeddings.
        values = (xi[:,i] if len(xi.dtype) == 0 else xi[xi.dtype.names[i]])
        if (len(i_map) > 1):
            ix = to_int(values, i_map)
            ix[ix > 0] += base
            _xi[:,i] = ix
            base += len(i_map)
    return _xi


# Convert all inputs to the AXY model into the expected numpy format.
def to_array(ax, axi, sizes, x, xi, y=None, yi=None, yw=None, maps=None):
    if (maps is None):
        maps = dict()
        axi_map = list()
        xi_map = list()
        yi_map = list()
        yi_embeddings = None
    else:
        axi_map = maps.get("axi_map", list())
        xi_map = maps.get("xi_map", list())
        yi_map = maps.get("yi_map", list())
        yi_embeddings = maps.get("yi_embeddings", None)
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
    else:                yi = np.zeros((nm,0), dtype="int64", order="C")
    if (yw is not None): yw = np.asarray(np.asarray(yw, dtype="float32").reshape((nm,-1)), order="C")
    else:                yw = np.zeros((nm,0), dtype="float32", order="C")
    if (x is not None): x = np.asarray(x, dtype="float32", order="C")
    else:               x = np.zeros((nm,0), dtype="float32", order="C")
    if (xi is not None): xi = np.asarray(xi)
    else:                xi = np.zeros((nm,0), dtype="int64", order="C")
    if (sizes is not None): sizes = np.asarray(sizes, dtype="int64")
    else:                   sizes = np.zeros(0, dtype="int64")
    na = sizes.sum()
    if (ax is not None): ax = np.asarray(ax, dtype="float32", order="C")
    else:                ax = np.zeros((na,0), dtype="float32", order="C")
    if (axi is not None): axi = np.asarray(axi)
    else:                 axi = np.zeros((na,0), dtype="int64", order="C")
    # Make sure that all inputs have the expected shape.
    assert (len(ax.shape) in {1,2}), f"Bad ax shape {ax.shape}, should be 1D or 2D matrix."
    assert (len(axi.shape) in {1,2}), f"Bad axi shape {axi.shape}, should be 1D or 2D matrix."
    assert (len(sizes.shape) == 1), f"Bad sizes shape {sizes.shape}, should be 1D int vector."
    assert (len(x.shape) in {1,2}), f"Bad x shape {x.shape}, should be 1D or 2D matrix."
    assert (len(xi.shape) in {1,2}), f"Bad xi shape {xi.shape}, should be 1D or 2D matrix."
    assert (len(y.shape) in {1,2}), f"Bad y shape {y.shape}, should be 1D or 2D matrix."
    assert (len(yi.shape) in {1,2}), f"Bad yi shape {yi.shape}, should be 1D or 2D matrix."
    assert (len(yw.shape) in {1,2}), f"Bad yw shape {yw.shape}, should be 1D or 2D matrix."
    # Reshape inputs to all be two dimensional (except sizes).
    if (len(ax.shape) == 1): ax = ax.reshape((-1,1))
    if ((len(axi.shape) == 1) and (len(axi.dtype) == 0)): axi = axi.reshape((-1,1))
    if (len(x.shape) == 1): x = x.reshape((-1,1))
    if (len(xi.shape) == 1) and (len(xi.dtype) == 0): xi = xi.reshape((-1,1))
    if (len(y.shape) == 1): y = y.reshape((-1,1))
    if (len(yi.shape) == 1) and (len(yi.dtype) == 0): yi = yi.reshape((-1,1))
    if (len(yw.shape) == 1): yw = yw.reshape((-1,1))
    assert (yw.shape[1] in {0, 1, y.shape[1]}), f"Bad yw shape {yw.shape}, should have 0 columns or 1 column{' or '+str(y.shape[1])+' columns' if (y.shape[1] > 1) else ''}."
    # Set the output size and numeric input sizes.
    mdo = y.shape[1]
    mdn = x.shape[1]
    adn = ax.shape[1]
    # Handle mapping "xi" into integer encodings.
    xi_cols = len(xi.dtype) or xi.shape[1]
    if (xi_cols > 0):
        assert (xi_map is not None), f"Provided data for 'xi' has {xi_cols} columns, 'xi_map' is None."
        if (len(xi_map) == 0):
            xi_map = i_map(xi, xi_map)
        else:
            assert (xi_cols == len(xi_map)), f"Bad number of columns in 'xi', {xi_cols}, expected {len(xi_map)} columns based on provided 'xi_map' map."
        xi = i_encode(xi, xi_map)
        mne = sum(map(len, xi_map))
    else: mne = 0
    # Handle mapping "axi" into integer encodings.
    axi_cols = len(axi.dtype) or axi.shape[1]
    if (axi_cols > 0):
        assert (axi_map is not None), f"Provided data for 'axi' has {axi_cols} columns, 'axi_map' is None."
        if (len(axi_map) == 0):
            axi_map = i_map(axi, axi_map)
        else:
            assert (axi_cols == len(axi_map)), f"Bad number of columns in 'axi', {axi_cols}, expected {len(axi_map)} columns based on provided 'axi_map' map."
        axi = i_encode(axi, axi_map)
        ane = sum(map(len, axi_map))
    else: ane = 0
    # Handle mapping "yi" into integer encodings.
    yi_cols = len(yi.dtype) or yi.shape[1]
    if (yi_cols > 0):
        assert (yi_map is not None), f"Provided data for 'yi' has {yi_cols} columns, 'yi_map' is None."
        if (len(yi_map) == 0):
            yi_map = i_map(yi, yi_map)
        else:
            assert (yi_cols == len(yi_map)), f"Bad number of columns in 'yi', {yi_cols}, expected {len(yi_map)} columns based on provided 'yi_map' map."
        yi = i_encode(yi, yi_map)
        yne = sum(map(len, yi_map))
    else: yne = 0
    # Handle mapping integer encoded "yi" into a single real valued y.
    if (yne > 0):
        # Use a regular simplex to construct equally spaced embeddings for the categories.
        if ((yi_embeddings is None) or (yi_embeddings.size == 0)):
            from tlux.math import regular_simplex
            yi_embeddings = regular_simplex(yne).astype("float32")
        else:
            assert (yi_embeddings.shape[1] == (yne-1)), f"Provided 'yi_embeddings' had shape {yi_embeddings.shape}, but expected a dimension of {yne-1} based on the number of categories."
        # Add a zero vector to the front for "unknown" outputs.
        embedded = np.concatenate((
            np.zeros((1,yne-1), dtype="float32"),
            yi_embeddings), axis=0)
        _y = np.zeros((nm, mdo+yne-1), dtype="float32")
        _y[:,:mdo] = y[:,:]
        for i in range(yi.shape[1]):
            _y[:,mdo:] += embedded[yi[:,i]]
        y = _y
        mdo += yne-1
    # Update all maps and return.
    maps.update(dict(
        axi_map = axi_map,
        xi_map = xi_map,
        yi_map = yi_map,
        yi_embeddings = yi_embeddings,
    ))
    # Return all the shapes and numpy formatted inputs.
    shapes = dict(
        mdn = mdn,
        mne = mne,
        mdo = mdo,
        adn = adn,
        ane = ane,
        yne = yne,
    )
    return nm, na, ax, axi, sizes, x, xi, y, yw, shapes, maps
