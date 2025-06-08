import multiprocessing
import queue
import threading
import time
import traceback


# Save "data" in a file titled "file_name" using pickle.
def save(data, file_name="_save.pkl"):
    try:
        import pickle
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    except:
        import dill as pickle
        with open(file_name, "wb") as f:
            pickle.dump(data, f)


# Load data from a pickle file titled "file_name".
def load(file_name="_save.pkl"):
    try:
        import pickle
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        return data
    except:
        import dill as pickle
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        return data


# Class for timing operations. Initialize to start, call to check, use
# the `start` and `stop` methods / attributes to set / observe values.
class Timer:
    #-----------------------------------------------------------------
    #                            Private
    _a = 0
    _b = None
    # Offer a function for explicitly beginning a timer.
    def _begin(self):
        self._a = time.time()
        self._b = None
        return self._a
    # End function for this timer.
    def _end(self):
        self._b = time.time()
        return self._b
    # Return the elapsed time since start.
    def _check(self):
        if   (self._a is None): return 0
        elif (self._b is None): return time.time() - self._a
        else:                   return self._b - self._a
    # Overwrite the "__call__" method of the provided object.
    def _callset(obj, func):
        class Float(type(obj)):
            def __call__(self): return func()
            def __repr__(self): return str(float(self))
        return Float(obj)
    #-----------------------------------------------------------------
    #                             Public
    # 
    # Initialization.
    def __init__(self): self._begin()
    # If not stopped, return elapsed time, otherwise return total time.
    def __call__(self, precision=2): return float(f"{self._check():.{precision-1}e}")
    def __str__(self): return str(self())
    # Return "start time" as attribute, "begin timer" as function.
    @property
    def start(self): return Timer._callset(self._a, self._begin)
    # Return "stop time" as attribute, "end timer, return total" as function.
    @property
    def stop(self):
        if (self._b == None): return Timer._callset(self._end(), self.total)
        else:                 return Timer._callset(self._b,     self.total)
    # Return the "total time from start" if running, return "total time" if finished.
    @property
    def total(self):
        if (self._b == None): return Timer._callset(self._check(), self._check)
        else:                 return Timer._callset(self._b - self._a, lambda: self._b - self._a)


# A decorator class for object types that make them behave asynchronously by
# housing instances in their own process. Returned values for all internal attributes
# and methods are 'queue.Queue' objects that have a '.get' method to explicitly
# retrieve results (or wait if the results are not ready yet).
#
# Example:
# 
#   async_obj_type = Asynchronous(SlowObject) # Create an asynchronous object.
#   obj = async_obj_type(*init_args, **init_kwargs)
#   async_result = obj.slow_method(*method_args, **method_kwargs) # Call a slow or expensive method.
#   ... # Do other operations without waiting for results.
#   result = async_result.get() # Retrieve the result from the object method call.
#
class Asynchronous:
    # The following are for the specification of this decorator type.
    _type: type = type(None)
    _stop_word: str = "__exit__"
    _verbose: bool = True
    # The following are populated for instantiated asynchronous objects of this type.
    _instance: object = None # Local instance for the asychronous jobs.
    _tasks: multiprocessing.Queue = None # Queue holding tasks.
    _results: multiprocessing.Queue = None # Queue holding results.
    _result_ids: dict = None # Dictionary holding identities of returned result queues.
    _type_tasks: multiprocessing.Queue = None # Queue holding type check tasks.
    _type_results: multiprocessing.Queue = None # Queue for type check results.
    _communicator: threading.Thread = None # Thread that communicates in shared memory space.
    _worker: multiprocessing.Process = None # Process that runs asynchronously.
    _type_checker: threading.Thread = None # Thread for type checking (in the asynchronous instance).
    _attributes_callable: dict = None # Dictionary holding attribute names and whether they're callable or not.

    # Initialization of a new Asynchronous type.
    def __init__(self, object_type=type(None), stop_word="__exit__", verbose=True):
        super().__setattr__("_type", object_type)
        super().__setattr__("_stop_word", stop_word)
        super().__setattr__("_verbose", verbose)

    # Wrap "Asynchronous" around the original type name.
    def __repr__(self): return f"<Asynchronous {self.__getattribute__('_type').__name__} instance at {id(self)}>"

    # Create an Asynchronous instance of the target object.
    def __call__(self, *args, **kwargs):
        worker = self.__getattribute__("_worker")
        # Presume this call is an initialization of an Asynchronous object.
        if (worker is None):
            obj = type(self)(
                object_type=self.__getattribute__("_type"),
                stop_word=self.__getattribute__("_stop_word"),
                verbose=self.__getattribute__("_verbose"),
            )
            obj.__getattribute__("_start")(*args, **kwargs)
            return obj
        # Presume this "__call__" was meant for the underlying object.
        else: return self.__getattr__("__call__")

    # Make sure all processes and threads are exited.
    def __del__(self):
        # Terminate the type checker thread (owned by the worker process).
        type_tasks = self.__getattribute__("_type_tasks")
        if (type_tasks is not None):
            type_tasks.put(None)
        # Terminate the worker.
        worker = self.__getattribute__("_worker")
        if ((worker is not None) and (worker.is_alive())):
            tasks = self.__getattribute__("_tasks")
            tasks.put((None, self.__getattribute__("_stop_word"), (), {}))
            worker.terminate()
        # Terminate the communicator thread (owned by the current process).
        communicator = self.__getattribute__("_communicator")
        if ((communicator is not None) and (communicator.is_alive())):
            results = self.__getattribute__("_results")
            results.put((None, None))
            communicator.join()

    # Set up an Asynchronous instance with appropriate processes and communication channels.
    def _start(self, *args, **kwargs):
        # These local dictionaries will hold the identifiers of the returned Queue
        # objects so they can be populated with results, and a cache dictating which
        # attributes of the asynchronous object instance are callable.
        super().__setattr__("_result_ids", {})
        super().__setattr__("_attributes_callable", {})
        # The following queues handle tasks, results, type tasks, and type results.
        super().__setattr__("_tasks", multiprocessing.Queue())
        super().__setattr__("_results", multiprocessing.Queue())
        super().__setattr__("_type_tasks", multiprocessing.Queue())
        super().__setattr__("_type_results", multiprocessing.Queue())
        # Create an asychronous worker that is independent (in memory and CPU).
        super().__setattr__("_worker", multiprocessing.Process(
            target=self.__getattribute__("_work"),
            daemon=True,
        ))
        self.__getattribute__("_worker").start()
        self.__getattribute__("_tasks").put((args, kwargs))
        # Create a thread (shared memory) that collects multiprocessed results and enqueues them.
        super().__setattr__("_communicator", threading.Thread(
            target=self.__getattribute__("_communicate"),
            daemon=True,
        ))
        communicator = self.__getattribute__("_communicator")
        communicator.start()
        # Get all of the attributes and their callability.
        attributes_callable = self.__getattribute__("_attributes_callable")
        type_results = self.__getattribute__("_type_results")
        attr_name, is_callable = type_results.get()
        while (attr_name is not None):
            attributes_callable[attr_name] = is_callable
            attr_name, is_callable = type_results.get()

    # This method is meant to be executed by a thread in the asynchronous process,
    # it can be used to check the type of the attributes of the instance.
    def _check_type(self):
        type_tasks = self.__getattribute__("_type_tasks")
        type_results = self.__getattribute__("_type_results")
        instance = self.__getattribute__("_instance")
        verbose = self.__getattribute__("_verbose")
        # On initialization, put all of the known attributes and their callability.
        for attr_name in dir(instance):
            type_results.put((attr_name, callable(getattr(instance, attr_name))))
        type_results.put((None, None))
        # Go into the listening loop, in case another attribute's callability status is needed.
        while True:
            attr_name = type_tasks.get()
            if (attr_name is None): break
            try: attr_type = callable(getattr(instance, attr_name))
            except Exception as exception: attr_type = exception
            type_results.put(attr_type)

    # Communicate the results of the asynchronous calls back into the local queues.
    # This method shares memory space with the parent process, so it can asynchronously
    # put results into the local 'queue.Queue' objects that were already returned.
    def _communicate(self):
        results = self.__getattribute__("_results")
        result_ids = self.__getattribute__("_result_ids")
        while True:
            identity, result = results.get()
            # When the identity is None, that means the communicator should exit.
            if identity is None:
                break
            # We have a result, put it into the appropriate Queue.
            else:
                result_ids.pop(identity).put(result)

    # Work on the assigned tasks from the parent process. This method will run
    # in its own process isolated from the parent by memory and CPU time.
    def _work(self):
        # Get the task queue.
        tasks = self.__getattribute__("_tasks")
        # Create a local instance of the (now asynchronous) type.
        args, kwargs = tasks.get()
        super().__setattr__("_instance", self.__getattribute__("_type")(*args, **kwargs))
        # Start the type checker thread.
        super().__setattr__("_type_checker", threading.Thread(
            target=self.__getattribute__("_check_type"),
            daemon=True,
        ))
        self._type_checker.start()
        # Get other local attributes that will be reused while working.
        results = self.__getattribute__("_results")
        instance = self.__getattribute__("_instance")
        verbose = self.__getattribute__("_verbose")
        while True:
            identity, name, args, kwargs = tasks.get()
            if identity is None: break
            # Get the attribute and do the appropriate action.
            try:
                attr = getattr(instance, name)
                # For callable attributes, execute the method.
                if callable(attr): results.put((identity, attr(*args, **kwargs)))
                # For value-only attributes, just return the value.
                else: results.put((identity, attr))
            # When an exception is raised, put it into the queue of results.
            except Exception as exception:
                if verbose: traceback.print_exc()
                results.put((identity, exception))
        # Terminate the type checker.
        type_checker = self.__getattribute__("_type_checker")
        type_checker.terminate()

    # This method captures all attempts to access an internal attribute of the instance
    # and redirects the request to the isolated process that the instance is running
    # inside of. If the specified attribute is callable, an asynchronous method is returned
    # that genereates 'queue.Queue' objects as results to calls. If the attribute is not
    # callable, then a 'queue.Queue' object is returned that will contain the value.
    def __getattr__(self, name):
        tasks = self.__getattribute__("_tasks")
        # Check to see if the instance attribute is callable.
        attributes_callable = self.__getattribute__("_attributes_callable")
        is_callable = attributes_callable.get(name, None)
        # If we do not already know the callability of the attribute, make a request to check.
        if (is_callable is None):
            type_tasks = self.__getattribute__("_type_tasks")
            type_results = self.__getattribute__("_type_results")
            type_tasks.put(name)
            is_callable = type_results.get()
            if isinstance(is_callable, Exception): raise(is_callable)
            else: attributes_callable[name] = is_callable
        # Fetch the queue that will hold the identities of the results.
        result_ids = self.__getattribute__("_result_ids")
        # If the attribute is callable, generate an asynchronous wrapper function and return it.
        if is_callable:
            def asynchronous_method(*args, **kwargs):
                async_result = queue.Queue()
                result_ids[id(async_result)] = async_result
                tasks.put((id(async_result), name, args, kwargs))
                return async_result
            return asynchronous_method
        # Else, this attribute just has a value and that value needs to be returned as a Queue.
        else:
            async_result = queue.Queue()
            result_ids[id(async_result)] = async_result
            tasks.put((id(async_result), name, tuple(), dict()))
            return async_result

    # Set an attribute on the asynchronous instance.
    def __setattr__(self, name, value):
        raise(NotImplementedError)
