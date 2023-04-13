# Tools for profiling.
import inspect
import gc
import os
import sys
import time
from functools import wraps
from collections import Counter


# External module.
try:
    import psutil
except ImportError:
    # Create a custom local version of 'psutil'. It will be slower, but it will work.
    SHOW_PSUTIL_WARNING = True
    class psutil:
        pid = 0
        rss_for_pid = "ps -p {pid} -o rss="
        def __init__(self):
            # Generate a warning about using a slower method for retrieving memory usage.
            global SHOW_PSUTIL_WARNING
            if (SHOW_PSUTIL_WARNING):
                import logging
                logging.warning("Using custom 'psutil' class that relies on calls to 'os.system'. It will be much slower than if you install 'psutil' with:\n  python3 -m pip install --user psutil\n")
                SHOW_PSUTIL_WARNING = False
            # Set the PID.
            import os
            self.pid = os.getpid()
            # Get the platform, if it Windows then update the command for getting memory utilization.
            import platform
            if platform.system() == 'Windows':
                self.rss_for_pid = "wmic process where ProcessID={pid} get WorkingSetSize"
        def Process():
            return psutil()
        def memory_info(self):
            return self
        @property
        def rss(self):
            # Execute the system command that gets the resident memory size of a process for this platform.
            mem_use_kb = os.popen(self.rss_for_pid.format(pid=self.pid)).read()
            # Return the memory usage in bytes as an integer.
            return int(mem_use_kb) * 1024


# Declare the divisors for different memory units.
MEM_UNIT_DIVISORS = {
    "KB" : 2**10,
    "MB" : 2**20,
    "GB" : 2**30,
    "TB" : 2**40,
}


# Return the memory usage currently of this process.
def mem_use(proc=psutil.Process(), unit="MB"):
    return proc.memory_info().rss / MEM_UNIT_DIVISORS[unit]


# Inspect the count and highest memory consuming objects.
def inspect_memory(top=10, max_history=1, history=[]):
    # Count the number of objects in memory and the total size.
    c = Counter(type(o).__name__ for o in gc.get_objects())
    s = sum(sys.getsizeof(o) for o in gc.get_objects())
    size_str   = f":: Total size: {s / (2**20):.2f}MB"
    top_str    = f":: Top obj counts: {c.most_common(top)}"
    # Measure the change in counts.
    if (len(history) > 0):
        d = sorted(((k, c.get(k,0) - history[-1].get(k,0)) for k in set(c)|set(history[-1])),
                   key = lambda i: (-i[1], i[0]))[:top]
        change_str = f":: Top count changes: {d}"
    else:
        change_str = ""
    width = max(len(size_str), len(top_str), len(change_str))
    # Print the update.
    print(":"*width)
    print(size_str)
    print(top_str)
    if (len(change_str) > 0): print(change_str)
    print(":"*width)
    # Update the history.
    history.append(c)
    while (len(history) > max_history):
        history.pop(0)


# Generate a summary string for a profile of a function.
def summarize_profile(profile, munit="MB", tunit="s", mdiv=2**20, tdiv=1):
    if len(profile) == 0:
        return "The function has not yet been executed."
    lines = []
    # Column widths.
    widths = [
        len("Time"),
        len("MDelta"),
        len("Mem"),
        len("Execs"),
        0,  # length of the code strings
    ]
    # Get the file ane function names.
    file_name, function_name, _, _ = next(iter(profile))
    # Get the totals for memory and time changes.
    true_time = 0
    time_total = 0
    mem_total = 0
    pos_mem_total = 0
    for (_, _, line, code), (times, mdelta, mem, execs) in profile.items():
        true_time += sum(times)
        if line != "call":
            time_total += sum(times)
            mem_total += sum(mdelta)
            pos_mem_total += sum((v for v in mdelta if v > 0))
    # Iterate over the lines of code and their stats.
    for (_, _, line, code), (times, mdelta, mem, execs) in sorted(
            profile.items(), key=lambda k: 0 if k[0][2] == "call" else int(k[0][2])
    ):
        # Collapse the lists to specific values.
        time = sum(times)
        mdelta = sum(mdelta)
        mem = (min(mem), max(mem))
        # Store the profiling details for this line.
        mem = f"[{mem[0]/mdiv:.2f}{munit}, {mem[1]/mdiv:.2f}{munit}]"
        if line == "call":
            time = ""
            mdelta = ""
        else:
            time = f"{time / tdiv:.2f}{tunit} ({100.0 * time / time_total if time_total > 0 else 0:.0f}%)"
            mdelta = f"{mdelta / mdiv:.2f}{munit} ({100.0 * mdelta / pos_mem_total if pos_mem_total > 0 else 0:.0f}%)"
        execs = f"{execs}"
        # Store the max of the column widths.
        widths[:] = (max(ov, len(nv)) for (ov, nv) in zip(
            widths, (time, mdelta, mem, execs, code)
        ))
        # Append the output to include in the string summary.
        lines.append((line, time, mdelta, mem, execs, code))
    # Generate the summary string.
    string = f"Line   {'Time':<{widths[0]}s}  {'MDelta':<{widths[1]}s}  {'Mem':<{widths[2]}s}  {'Execs':<{widths[3]}s}  Code"
    bar = "-" * (len(string) + 4) + "\n"
    string = (
        "_" * (len(string) + 4)
        + "\n"
        + f"File:     {file_name}\n"
        + f"Function: {function_name}\n"
        + f"Memory:   {mem_total / mdiv:.2f}{munit}\n"
        + f"Time:     {true_time / tdiv:.2f}{tunit} ({time_total / tdiv:.2f}{tunit} accounted for)\n"
        + string + "\n"
        + bar
    )
    for (number, time, mdelta, mem, execs, code) in lines:
        string += f"{number:5s}  {time:{widths[0]}s}  {mdelta:{widths[1]}s}  {mem:{widths[2]}s}  {execs:{widths[3]}s}  {code:<{widths[4]}s}\n"
    # Add one last line.
    string += bar
    return string


# A decorator that can be used to show the line-by-line memory and time consumtion of a python function.
def profile(f):
    # Store some local variables that can be reused in the scope of the profiling decorator.
    process = psutil.Process()
    profile = {}
    previous = dict(
        line = "call",
        code = "",
        time = time.time(),
        mem = 0,
        depth = 0,
    )
    # Create a function that monitors memory usage and elapsed time between executed lines of python code.
    # Arguments include:
    #   frame - the frame object (from insepct or sys.trace);
    #   event - one of 'call', 'exception', 'line', 'opcode', or 'return';
    #   arg - something that we do not use here.
    def track(frame, event, arg):
        nonlocal process, profile, previous
        # Depth increments whenever a call is made and we can use this to block tracking in child functions.
        if event == "call": previous['depth'] += 1
        # When we encounter a line that is in the target function, we record stats.
        if (previous['depth'] == 1) and (event in {"call", "line", "return"}):
            # Store the time delta (since last line).
            elapsed = time.time() - previous["time"]
            # Get information about the current scope.
            frame_info = inspect.getframeinfo(frame)
            file_name = os.path.join(
                os.path.basename(os.path.split(frame_info.filename)[0]),
                os.path.basename(frame_info.filename)
            )
            function_name = frame_info.function
            line, code = previous["line"], previous["code"]
            # Create a hashable identifier for this line.
            line_id = (file_name, function_name, line, code)
            # Retreive / create profiling information for this line.
            profile_vals = profile.get(line_id, [[], [], [], 0])  # [time, mem delta, mem, executions]
            # Get and update histories.
            mem = process.memory_info().rss
            profile_vals[0].append(elapsed)
            profile_vals[1].append(mem - previous["mem"])
            profile_vals[2].append(mem)
            profile_vals[3] += 1
            # Store the updated history for this line.
            profile[line_id] = profile_vals
            # Update the "code" and "line number" for the next iteration.
            previous["code"] = ("".join(frame_info.code_context)).rstrip().replace("\n", "; ")
            previous["line"] = str(frame.f_lineno)
            previous["mem"] = mem
            previous["time"] = time.time()
        # When a return event occurs, decrement the depths. Also empty history each time the function fully exits.
        if event == "return":
            previous["depth"] -= 1
            if previous["depth"] == 0:
                previous["code"] = ""
                previous["line"] = "call"
        # Trace fuctions return the "next" trace function, in this case, itself (to be called again later).
        return track
    # 
    # Now we create a function that wraps the original function with added tracking.
    @wraps(f)
    def profiled(*args, **kwargs):
        nonlocal f
        try:
            # Get any existing trace (in case multiple python debuggers are executing).
            existing_trace = sys.gettrace()
            if existing_trace is None:
                # Assign the memory tracking function as a trace function (to be called by the Python interpreter).
                sys.settrace(track)
            else:
                # Generate a new trace function that executes this memory trace and the existing one.
                def stacked_trace(*args, **kwargs):
                    nonlocal track, existing_trace
                    track = track(*args, **kwargs)
                    existing_trace = existing_trace(*args, **kwargs)
                    return stacked_trace

                # Set the trace to the new stacked tracing function.
                sys.settrace(stacked_trace)
            # Execute the original (now decorated) function.
            result = f(*args, **kwargs)
        # Reset the trace to its state before this decorator.
        finally:
            sys.settrace(existing_trace)
        return result
    # 
    profiled.nonlocals = {
        "function": f,
        "process": process,
        "profile": profile,
        "previous": previous,
    }
    profiled.profile_summary = lambda: summarize_profile(profile)
    profiled.show_profile = lambda: print(summarize_profile(profile))
    # 
    return profiled

