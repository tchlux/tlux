import time
from tlux.search.hkm.jobs import spawn_job

# Define a simple function to be executed as a job.
def _example_add(x: int, y: int) -> int:
    # Returns the sum of x and y after a short delay.
    time.sleep(0.05)
    print(f"sum={x+y}")
    return 0

# Define a simple function to be executed as a job.
def _example_fail(x: int, y: int) -> int:
    # Returns the sum of x and y after a short delay.
    time.sleep(0.05)
    print(f"sum={x+y}")
    raise ValueError(f"This *should* fail! {time.ctime()}")
    return 0

# Define a simple function that theoretically waits on an upstream.
def _example_dep(x: int) -> int:
    # Multiplies x by 2 after a short delay.
    time.sleep(0.05)
    print(f"dep_result={x * 2}")
    return 0

# Test that a failing job completes and produces error output.
#
# Returns:
#   None
#
def test_failed_job() -> None:
    import time
    job = spawn_job("tests.test_jobs._example_fail", 7, 8)
    print(f"Spawned job ID: {job.id}")
    job.wait_for_completion(poll_interval=0.02)
    print("Job stdout:", repr(job.stdout))
    print("Job stderr:", repr(job.stderr))
    print(f"Job finished? {not job.is_running()}")
    print(f"Job exit code: {job.exit_code}")
    print()
    print("STDERR")
    print("-"*100)
    print(job.stderr.strip())
    print("-"*100)

# Test that a successful job completes and produces correct output.
#
# Returns:
#   None
#
def test_successful_job() -> None:
    import time
    job = spawn_job("tests.test_jobs._example_add", 7, 8)
    print(f"Spawned job ID: {job.id}")
    while job.is_running():
        time.sleep(0.02)
    print("Job stdout:", repr(job.stdout))
    print("Job stderr:", repr(job.stderr))
    print(f"Job finished? {not job.is_running()}")
    print(f"Job exit code: {job.exit_code}")

# Test a job with a dependency; verifies that downstream job waits for upstream to finish.
#
# Returns:
#   None
#
def test_job_with_dependency() -> None:
    import time
    job_a = spawn_job("tests.test_jobs._example_add", 2, 3)
    job_b = spawn_job("tests.test_jobs._example_dep", 5, dependencies=[job_a])
    print(f"Spawned job_a: {job_a.id}, job_b: {job_b.id}")
    # Wait for job_a to complete.
    job_a.wait_for_completion(poll_interval=0.02)
    # Wait for job_b to complete.
    job_b.wait_for_completion(poll_interval=0.02)
    print("job_a stdout:", repr(job_a.stdout))
    if job_b:
        print("job_b stdout:", repr(job_b.stdout))


if __name__ == "__main__":
    test_successful_job()
    test_failed_job()
    test_job_with_dependency()
