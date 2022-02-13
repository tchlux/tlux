import numpy as np


# Given "d" categories that need to be converted into real space,
# generate a regular simplex in (d-1)-dimensional space. (all points
# are equally spaced from each other and the origin) This process
# guarantees that all points are not placed on a sub-dimensional
# manifold (as opposed to "one-hot" encoding).
def regular_simplex(num_categories):
    assert num_categories >= 1, f"Number of categories must be an integer greater than 0, received {num_categories}."
    # Special cases for one and two categories
    if num_categories == 1:
        return np.asarray([[0.]])
    elif num_categories == 2:
        return np.asarray([[0.],[1.]])
    # Standard case for >2 categories
    d = num_categories
    # Initialize all points to be zeros.
    points = np.zeros((d,d-1))
    # Set the initial first point as 1
    points[0,0] = 1
    # Calculate all the intermediate points
    for i in range(1,d-1):
        # Set all points to be flipped from previous calculation while
        # maintaining the angle "arcos(-1/d)" between vectors.
        points[i:,i-1] = -1/(d-i) * points[i-1,i-1]
        # Compute the new coordinate using pythagorean theorem
        points[i,i] = (1.0 - (points[i,:i]**2).sum())**(0.5)
    # Set the last coordinate of the last point as the negation of the previous
    points[i+1,i] = -points[i,i]
    # Return the regular simplex
    return points


# Generic class definition for creating an algorithm that can perform
# regression in the multi-variate setting. In these classes 'x' refers
# to a potentially multi-dimensional set of euclydian coordinates with
# associated single-dimensional response values referred to as 'y'.
# 
# The "fit" function is given both 'x' and associated 'y' and creates
# any necessary (or available) model for predicting 'y' at new 'x'.
# 
# The "predict" function predicts 'y' at potentially novel 'x' values.
# 
# Subclasses are expected to implement _fit and _predict methods which
# take 2D numpy arrays of points as input.
class Approximator:
    _y_shape = 2
    _classifier = False

    # By default, return the 
    def __str__(self):
        return type(self).split("'")[1] + " object"

    # These functions should be defined by child classes.
    def _fit(*args, **kwargs): raise(NotImplemented())
    def _predict(*args, **kwargs): raise(NotImplemented())
    def _gradient(*args, **kwargs): raise(NotImplemented())

    # Fit a 2D x and a 2D y with this model.
    def fit(self, x, y, xi=None, *args, **kwargs):
        # Sanity check the input.
        assert issubclass(type(x), np.ndarray), f"Expected 'x' to be a subclass of 'np.ndarray', received {type(x)}."
        assert len(x.shape) == 2, f"Expected 'x' to be a 2D array, received shape {x.shape}."
        assert issubclass(type(y), np.ndarray), f"Expected 'y' to be a subclass of 'np.ndarray', received {type(y)}."
        assert (x.shape[0] == y.shape[0]), f"First dimension of 'x' ({x.shape[0]}) and 'y' ({y.shape[0]}) do not match."
        if (xi is not None):
            assert issubclass(type(xi), np.ndarray), f"Expected 'xi' to be a subclass of 'np.ndarray', received {type(x)}."
            assert len(xi.shape) == 2, f"Expected 'xi' to be a 2D array, received shape {xi.shape}."
            assert (x.shape[0] == xi.shape[0]), f"First dimension of 'x' ({x.shape[0]}) and 'xi' ({xi.shape[0]}) do not match."
            kwargs["xi"] = xi
        # Check shape of y (allowed to be 1D or 2D). Coerce into 2D.
        self._y_shape = len(y.shape)
        if (len(y.shape) == 1):
            y = y.reshape((-1,1))
        # Check to see if this should be a classifier (because output is not continuous).
        if (not np.issubdtype(y.dtype, np.floating)):
            self._classifier = True
            self._categories = []
            self._embeddings = []
            # Generate embeddings for all categorical values.
            output_size = 0
            for i in range(y.shape[1]):
                unique_values = np.unique(y[:,i])
                embeddings = np.identity(len(unique_values))
                self._categories.append( unique_values )
                self._embeddings.append( embeddings )
                output_size += len(unique_values)
            # Map the provied categorical values into embeddings.
            yi = y
            y = np.zeros((y.shape[0], output_size))
            col_start = 0
            for i in range(len(self._categories)):
                size = len(self._categories[i])
                col_end = col_start + size
                indices = yi[:,i] == self._categories[i]
                y[:,col_start:col_end] = self._embeddings[i][indices]
                col_start = col_end
        # Apply the fit function.
        return self._fit(x, y, *args, **kwargs)
        
    # Predict y at new x locations from fit model.
    def predict(self, x, *args, **kwargs):
        # Sanity check the input.
        assert issubclass(type(x), np.ndarray), f"Expected 'x' to be a subclass of 'np.ndarray', received {type(x)}."
        # Allow for single-point prediction.
        single_response = (len(x.shape) == 1)
        if single_response:
            x = x.reshape((1,x.size))
        # Produce predictions.
        y = self._predict(x, *args, **kwargs)
        # Return categories back to their format given during fit.
        if (self._classifier):
            yi = []
            col_start = 0
            for i in range(len(self._categories)):
                size = len(self._categories[i])
                col_end = col_start + size
                indices = np.argmax(y[:,col_start:col_end], axis=1)
                yi.append( self._categories[i][indices] )
                col_start = col_end
            y = np.asarray(yi)
        # Reduce output to expected shape.
        if (self._y_shape == 1) or single_response:
            y = y.reshape((-1,))
        # Return prediction.
        return y

    # Give the model gradient at new x locations from fit model.
    def gradient(self, x, *args, **kwargs):
        # Sanity check the input.
        assert issubclass(type(x), np.ndarray), f"Expected 'x' to be a subclass of 'np.ndarray', received {type(x)}."
        assert not self._classifier, "The gradient with respect to categorical outputs is undefined."
        # Allow for single-point prediction.
        single_response = (len(x.shape) == 1)
        if single_response:
            x = x.reshape((1,x.size))
        # Produce predictions.
        y = self._gradient(x, *args, **kwargs)
        # Reduce output to expected shape.
        if (self._y_shape == 1) or single_response:
            y = y.reshape((-1,))
        # Return prediction.
        return y

    # Wrapper for 'predict' that returns a single value for a single
    # prediction, or an array of values for an array of predictions
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)



