from typing import List

# Function for creating a downsampling plan for sequences that are too long. This
#  uses symmetric exponential weighting applied to the elements in the sequence,
#  where elements at the front and back are exponentially more likely to be kept
#  than elements in the middle. An optional "bias" term can be set to 0.0 to use
#  linear sampling, 1.0 for pure exponential, and larger values to prefer edges.
# 
# Returns a list of indices that should be "kept" from the input sequence
#  according to this downsampling procedure.
# 
def downsample_indices_symmetric_exponential(
    sample_size: int,
    total: int,
    bias: float = 2.0
) -> List[int]:
    import numpy as np
    # If we can keep all indices, then do that.
    if total == 0:
        return []
    elif sample_size >= total:
        return list(range(total))
    elif sample_size == 1:
        return [total // 2]
    # Otherwise a sampling strategy will need to be used.
    #
    # Make a quadratic with range [1, 2]
    quadratic = 1.0 + (1 - np.linspace(0, 2, (sample_size-1)))**2
    # Make an exponential with values in the range [0, 1] that are the desired "weights" of selection.
    exponential = np.exp(quadratic ** bias)
    exponential /= abs(exponential).max()
    # Compute the cumulative weight of selecting indices (inverse here since we round to do selection).
    cumulative = np.concatenate(([0.0], np.cumsum(1 / exponential)))
    cumulative /= abs(cumulative).max()
    # Get the initial set of chosen indices (might contain duplicates).
    indices = np.floor(cumulative * (total-1)).astype(int)
    # Compute the difference between adjacent indices.
    diff = np.diff(indices)
    # Compute the midpoint.
    mid = int(np.ceil(sample_size / 2))
    # Walk forwards, crowding out duplicates in the lower indices.
    i = -1
    while i < mid:
        i += 1
        while (i < len(diff)) and (diff[i] <= 0):
            indices[i+1] = indices[i] + 1  # increment the next index
            diff[i] = 1  # diff to next is now 1
            i += 1  # hop to next
            if i < len(diff):  # update the diff preceding changed value
                diff[i] = indices[i+1] - indices[i]
    # Walk backwards, crowding out duplicates in the higher indices.
    i = sample_size
    while i > mid:
        i -= 1
        while (i >= 0) and (diff[i-1] <= 0):
            indices[i-1] = indices[i] - 1  # decrement the previous index
            diff[i-1] = 1  # diff from previous is now 1
            i -= 1  # hop to previous
            if i >= 0:  # update the diff preceding changed value
                diff[i-1] = indices[i] - indices[i-1]
    # Return the (now unique) indices that have been selected.
    return indices


# By default, when there is a "__main__", run tests.
if __name__ == "__main__":

    # Testing function.
    def _test_downsample_indices_symmetric_exponential():
        # Run tests to verify that the samples are at least "valid".
        for total in range(1, 128+1):
            for sample in range(1, total+1):
                indices = downsample_indices_symmetric_exponential(sample, total)
                if (
                    (len(set(indices)) < len(indices))  # all unique indices
                    or ((sample > 1) and (len({0,total-1} & set(indices)) < 2))  # first and last index are always included
                    or ((sample >= 3) and ((total % 2) == 1) and (sample % 1 == 1) and (total//2 not in indices))  # middle element is always included for odd sizes
                ):
                    print()
                    print("-"*100)
                    print("FAILURE")
                    print(" total size ", total)
                    print(" sample size", sample)
                    print()
                    print("Indices:")
                    print("", indices)
                    print()
                    print("-"*100)
                    print(flush=True)
                    raise(RuntimeError(f"Encountered failure. See printed logs."))

    print()
    print("Testing 'downsample_indices_symmetric_exponential'..", flush=True)
    _test_downsample_indices_symmetric_exponential()
    print(" passed.", flush=True)
