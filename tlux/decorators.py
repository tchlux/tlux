from typing import (
    Any,
    Callable,
    Optional,
    Union,
)


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



# ==================================================================
#                         AutoCLI Decorator
# 
# Decorator that extracts the documentation and arguemnts of a function
# (specified as input) and generates a commensurate command line
# interface. The expected format of the documentation for parsing is to
# include a first a few lines generally summarizing the behavior of the
# function. Then in the lines that follow arguments must be specified 
# as something of the form:
# 
#   argument_name -- Description of argument that
#                    may span multiple lines.
# 
# These comments about the argument will be extracted. The separator may
# be either ":", "-", or "--". Any of those will be captured correctly.
# 
# OPTIONAL ARGUMENTS:
#   scope -- The scope in which the 'main' function should be executed,
#            defaults to the typical "__main__".
#   trigger -- A trigger phrase that must be the first command line argument
#              after the executed file name in order for the 'main' to be
#              executed (allows for multiple 'auto_cli' in a single file).
#   wait_to_call -- True if a thread should be spawned that waits for the
#                   containing file to finish loading before executing 'main'.
#                   This is useful in cases where there are circular dependencies.
#   max_wait_seconds -- The maximum number of seconds to wait for the above action.
#   auto_main -- True if the 'main' should be automatically executed on decoration.
#                If False, then a manual 'if __name__ == "__main__": <func>.main()'
#                will be required to be written by the user.
# 
# USAGE: 
# 
#   # Description of function purpose.
#   #   arg_1 (str): The contents of the first argument.
#   @auto_cli
#   def <function_to_decorate>(...):
#      ...
# 
#   OR
# 
#   <function> = auto_cli(<function_to_decorate>)
# 
def auto_cli(
    scope: Union[str, Callable[..., Any]] = "__main__",
    trigger: Optional[str] = None,
    wait_to_call: bool = False,
    max_wait_seconds: int = 10,
    auto_main: bool = True
) -> Callable[..., Any]:
    import argparse
    import inspect
    import logging
    import re
    import sys
    import threading
    import time
    from functools import wraps
    from typing import (Dict, get_args, get_origin, List, Set, Tuple, TypeVar)

    # Handle two modes of usage, with and without being called within the
    #  decoration (e.g., "@auto_cli", "@auto_cli()", "@auto_cli(trigger='abc')").
    if type(scope) is str:
        func = None
    else:
        func = scope
        scope = "__main__"

    # Get the __name__ of the calling scope.
    caller_scope: str = inspect.stack()[1][0].f_globals["__name__"]
    caller_file: str = inspect.stack()[1][0].f_globals.get("__file__", "N/A")
    logging.warning("\n")
    logging.warning(f"Caller file:  {repr(caller_file)}")
    logging.warning(f"Caller scope: {repr(caller_scope)}")
    logging.warning(f"CLI scope:    {repr(scope)}")
    if trigger is not None:
        logging.warning(f"CLI trigger:  {repr(trigger)}")

    # Define a type variable capturing the callable's signature
    Func = TypeVar("Func", bound=Callable[..., Any])

    # Check if a type is "Optional". If so, return the set of other types allowed. If not, return False.
    def get_optional_types(typ: type) -> Set[type]:
        # Get the origin of the type (e.g., Union for Optional types)
        origin = get_origin(typ)
        # If this is not a Union, it cannot be an Optional type.
        if origin is not Union:
            return set()
        # Check to see if "NoneType" is in the set of allowed types.
        elif not any(arg is type(None) for arg in get_args(typ)):
            return set()
        else:
            # Get the set of allowed types that are NOT "NoneType".
            return {arg for arg in get_args(typ) if not (arg is type(None))}

    # Extract the description of arguments from the comments and docstring of a function.
    def extract_arg_descriptions(
        comments: str, docstring: str
    ) -> Tuple[str, Dict[str, str]]:
        # Remove all preceding "#" and replace them with " " from the comments.
        comments: str = re.sub(
            "(\\s*)(#+)(\\s*)",
            lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3),
            comments,
        )
        # Generate the full function description.
        all_docs: str = comments + "\n" + docstring
        arg_descriptions: Dict[str, str] = {}
        arg_regex: str = r"^(\s*#?\s*)([a-zA-Z0-9_]+)(\s*)([(][^)]+[)])?(\s*)(:|-{1,2})(.*)$"
        arg_desc_line: str = r"^(\s*#?\s*)(.+)"
        previous_arg: str = ""
        function_description: str = ""
        beginning = True
        for line in all_docs.split("\n"):
            match_arg = re.search(arg_regex, line)
            match_desc = re.search(arg_desc_line, line)
            # If this looks like an argument, then record the description.
            if match_arg:
                beginning = False
                _, aname, _, atype, _, asep, adesc = match_arg.groups()
                if aname and adesc:
                    arg_descriptions[aname] = adesc.strip()
                    previous_arg = aname
                # No match, end of any previous argument.
                else:
                    previous_arg = ""
            # If this is probably a continuation of the previous argument, append it.
            elif match_desc and previous_arg:
                _, adesc = match_desc.groups()
                if adesc and adesc.strip():
                    arg_descriptions[previous_arg] = (
                        arg_descriptions[previous_arg] + " " + adesc.strip()
                    )
                # No matches, end of previous argument.
                else:
                    previous_arg = ""
            elif match_desc and beginning:
                _, adesc = match_desc.groups()
                if adesc:
                    function_description += " " + adesc
                else:
                    beginning = False
            # No matches, end of any previous argument.
            else:
                previous_arg = ""
        # Return the dictionary of *POTENTIAL* argument descriptions (might include extra noise).
        return function_description.strip(), arg_descriptions

    # Given a function description, signature, and dictionary of argument
    #  descriptions, define an argument parser for the function.
    def define_argument_parser(
        function_description: str,
        function_signature: inspect.Signature,
        arg_descriptions: Dict[str, str],
    ) -> argparse.ArgumentParser:
        # Define a parser for the function.
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description=function_description
        )
        for arg_name, arg_param in function_signature.parameters.items():
            # Get the type.
            arg_type = (
                str if arg_param.annotation is inspect._empty else arg_param.annotation
            )
            arg_type_str = repr(arg_type)
            if arg_type_str.startswith("<class '") and arg_type_str.endswith("'>"):
                arg_type_str = arg_type_str[len("<class '") : -len("'>")]
            if arg_type_str.startswith("typing."):
                arg_type_str = arg_type_str[len("typing.") :]
            # Verify the arguments are supported types.
            assert (
                arg_type in {str, int, float, Optional[str], Optional[int], Optional[float]}
            ) or (
                arg_type_str.startswith("Tuple[")
                and arg_type_str.endswith(", ...]")
                and (arg_type_str.count(",") == 1)
            ), f"\n  Unsupported argument type for automatic CLI:\n    {arg_type}\n    {repr(arg_type)}"
            # Define argument parameters.
            param_args = [f"--{arg_name}"]
            param_kwargs = {
                "type": arg_type,
                "help": f"({arg_type_str}) " + arg_descriptions.get(arg_name, ""),
            }
            # Make optional types not be required (and adjust type to be the base type).
            optional_types: Set[type] = get_optional_types(arg_type)
            if optional_types:
                if len(optional_types) > 1:
                    raise ValueError(
                        f"Cannot handle multiple optional types for argument {arg_name}."
                    )
                arg_type = next(iter(optional_types))
                param_kwargs["type"] = arg_type
                param_kwargs["required"] = False
                param_kwargs["default"] = None
            # Handle whether or not there is a default value.
            if arg_param.default is inspect._empty:
                if not optional_types:
                    param_kwargs["required"] = True
            else:
                param_kwargs["default"] = arg_param.default
                param_kwargs["required"] = False
            # Handle arguments that accept multiple elements.
            if "..." in arg_type_str:
                param_kwargs["type"] = str
                param_kwargs["nargs"] = "+"

                def typed_tuple(base_type: type) -> object:
                    class ParseTupleAction(argparse.Action):
                        def __call__(self, parser, namespace, values, option_string=None):
                            setattr(namespace, self.dest, tuple(map(base_type, values)))

                    return ParseTupleAction

                param_kwargs["action"] = typed_tuple(arg_type.__args__[0])
            # Add the argument.
            parser.add_argument(*param_args, **param_kwargs)
        # Return the constructed argument parser.
        return parser

    # Define the decorator.
    def decorator(func: Func) -> Func:
        def main() -> None:
            nonlocal trigger
            # Get the descriptions (help text) for all arguments from comments and/or docstring.
            function_description: str = ""
            arg_descriptions: Dict[str, str] = {}
            function_description, arg_descriptions = extract_arg_descriptions(
                inspect.getcomments(func) or "",
                inspect.getdoc(func) or "",
            )
            # Get the function signature (which has the type annotations and default values).
            function_signature = inspect.signature(func)
            # If this is the main execution thread, have a thread monitor to see if
            #  this function is being called. If it is, parse the command line arguments
            #  and call the function with those arguments.
            if (caller_scope == scope) and (
                (trigger is None) or (trigger in " ".join((sys.argv + [""])[:2]))
            ):
                # For custom trigger phrases, remove the trigger.
                if (
                    (trigger is not None)
                    and (len(sys.argv) > 1)
                    and (trigger in sys.argv[1])
                ):
                    sys.argv.pop(1)
                # Define a parser for the function.
                parser: argparse.ArgumentParser = define_argument_parser(
                    function_description, function_signature, arg_descriptions
                )
                # Parse all arguments.
                args: argparse.Namespace = parser.parse_args()
                args_str: str = "\n ".join(
                    f"{repr(key)}: {repr(value)}"
                    for (key, value) in dict(args._get_kwargs()).items()
                )
                logging.warning(f"CLI calling function with arguments:\n {args_str}\n\n")
                # Call "func" with the arguments.
                if not wait_to_call:
                    func(**dict(args._get_kwargs()))
                else:

                    # Define a function that will loop until this file is loaded.
                    def call_func_once_loaded() -> None:
                        nonlocal max_wait_seconds
                        sleep_start: float = time.time()
                        while hasattr(sys.modules["__main__"], "__cached__") and (
                            (time.time() - sleep_start) < max_wait_seconds
                        ):
                            time.sleep(0.1)
                        func(**dict(args._get_kwargs()))

                    # Call "func" with the arguments in a thread to let this file finish loading first.
                    t = threading.Thread(target=call_func_once_loaded)
                    t.start()

        # Call the "main" once decoration is complete if that is desired.
        if auto_main: main()

        # Define and return the decorated function.
        @wraps(func)
        def decorated_func(*args: List[Any], **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        decorated_func.main = main
        return decorated_func

    # Return the decorator.
    if func is None:
        return decorator
    else:
        return decorator(func)

