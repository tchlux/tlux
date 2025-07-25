import time


# Define a simple function to be executed as a job.
def _example_add(x: int, y: int) -> int:
    # Returns the sum of x and y after a short delay.
    time.sleep(1)
    print(f"sum={x+y}")
    return 0

# Define a simple function to be executed as a job.
def _example_fail(x: int, y: int) -> int:
    # Returns the sum of x and y after a short delay.
    time.sleep(1)
    print(f"sum={x+y}")
    raise ValueError(f"That's not the result I expected {time.ctime()}")
    return 0

