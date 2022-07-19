
# Class for timing operations. Initialize to start, call to check, use
# the `start` and `stop` methods / attributes to set / observe values.
class Timer:
    #-----------------------------------------------------------------
    #                            Private
    import time as _time
    _a = 0
    _b = None
    # Offer a function for explicitly beginning a timer.
    def _begin(self):
        self._a = self._time.time()
        self._b = None
        return self._a
    # End function for this timer.
    def _end(self):
        self._b = self._time.time()
        return self._b
    # Return the elapsed time since start.
    def _check(self):
        if   (self._a is None): return 0
        elif (self._b is None): return self._time.time() - self._a
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
    # If not stopped, return elapsed time, otherwise return total time.
    def __call__(self, precision=2): return float(f"{self._check():.{precision-1}e}")

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
