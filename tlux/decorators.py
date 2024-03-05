# ==================================================================
#                         SameAs Decorator
# 
# Decorator that copies the documentation and arguemnts of another
# function (specified as input). Useful for making decorators (:P)
# Optional "mention_usage" updates documentation when True to disclose
# the name of the function being wrapped (and the reuse of signature).
# 
# USAGE: 
# 
#   @same_as(<func_to_copy>)
#   def <function_to_decorate>(...):
#      ...
# 
#   OR
# 
#   <function> = same_as(<func_to_copy>)(<function_to_decorate>)
#   
def same_as(to_copy, mention_usage=False):
    import inspect
    # Create a function that takes one argument, a function to be
    # decorated. This will be called by python when decorating.
    def decorator_handler(func):
        if hasattr(func, "__name__"): original_name = func.__name__
        else:                         original_name = str(func)
        # Set the documentation string for this new function
        documentation = inspect.getdoc(to_copy)
        if documentation == None: 
            documentation = inspect.getcomments(to_copy)
        # Store the documentation and signature into the wrapped function
        if hasattr(to_copy, "__name__"):
            func.__name__ = to_copy.__name__
            if mention_usage:
                documentation = (
                    "\nThe function '%s' has been decorated with the signature "+
                    "of '%s'. (likely for aliasing / decoration)\n\n")%(
                        original_name, to_copy.__name__) + documentation
        # Try copying the signature if possible
        try:               func.__signature__ = inspect.signature(to_copy)
        except ValueError: pass
        # Finalize by copying documentation
        func.__doc__ = documentation
        return func
    # Return the decorator handler
    return decorator_handler


# ==================================================================
#                    "Cache in File" Decorator     
# 
# This decorator (when wrapped around a function) uses a hash of the
# string represenetation of the parameters to a function call in order
# to write a pickle file that contains the inputs to the function and
# the outputs to the function.
# 
# 
# USAGE: 
# 
#   @cache(<max_files>=10, <cache_dir>=os.curdir, <file_prefix>="Cache_[func.__name__]")
#   def <function_to_decorate>(...):
#      ...
# 
#   OR
# 
#   <function> = cache(<max_files>, <cache_dir>, <file_prefix>)(<function_to_decorate>)
#   
def cache(max_files=10, cache_dir=None, file_prefix=None, use_dill=False):
    import os, hashlib
    # Import "dill" if it is available, otherwise use pickle.
    if use_dill:
        try:    import dill as pickle
        except: import pickle
    else:
        import pickle    
    # Check to see if a cache directory was provided
    if (type(cache_dir) == type(None)): cache_dir = os.path.abspath(os.curdir)
    if (not os.path.exists(cache_dir)): os.makedirs(cache_dir)
    # Create a function that takes one argument, a function to be
    # decorated. This will be called by python when decorating.
    def decorator_handler(func):
        cache_prefix = file_prefix
        if (type(file_prefix) == type(None)):
            cache_prefix = "Cache_[%s]"%(func.__name__)
        def new_func(*args, **kwargs):
            # Identify a cache name via sha256 hex over the serialization
            hash_value = hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
            cache_suffix = ".pkl"
            cache_path = os.path.join(cache_dir, cache_prefix+"_"+hash_value+cache_suffix)
            # Check to see if a matching cache file exists
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    args, kwargs, output = pickle.load(f)
            else:
                # Calculate the output normally with the function
                output = func(*args, **kwargs)
                # Identify the names of existing caches in this directory
                existing_caches = [f for f in os.listdir(cache_dir)
                                   if cache_prefix in f[:len(cache_prefix)]]
                # Only save a cached file if there are fewer than "max_files"
                if len(existing_caches) < max_files:
                    with open(cache_path, "wb") as f:
                        pickle.dump((args, kwargs, output), f)
            # Return the output (however it was achieved)
            return output
        # Return the decorated version of the function with identical documntation
        return same_as(func)(new_func)
    # Return a function that is capable of decorating
    return decorator_handler
