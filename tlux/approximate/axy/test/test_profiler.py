import fmodpy

# Import the profiler.
profiler = fmodpy.fimport("../axy_profiler.f90", verbose=False, build_dir="test_axy_profiler", name="test_axy_profiler").profiler

# Run a simple test with a sleep command.
def test_profiler():
    print("test profiler ", end="...", flush=True)
    import time
    profiler.start_profiling("time_test")
    time.sleep(1)
    profiler.stop_profiling("time_test")
    prof = profiler.get_profile("time_test")
    assert (prof.wall_time > 1)
    assert (prof.cpu_time > 0)
    print(" passed.", flush=True)

if __name__ == "__main__":
    test_profiler()
