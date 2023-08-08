# Random points within a [0,1] box.
def box(num_points, dimension):
    from numpy.random import uniform
    return uniform(size=(num_points, dimension))


# Generate "num_points" random points in "dimension" that have uniform
# probability density over the unit ball scaled by "radius" (length of
# points are in range [0, "radius"]).
def ball(num_points, dimension, inside=True, radius=1.0):
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # If inside, then generate a random radius with probability
    #  proportional to the surface area of a ball with a given radius.
    if (inside): random_directions *= random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * random_directions.T


# Wrapper for the "random.ball" function with no points inside.
def sphere(num_points, dimension, radius=1.0):
    return ball(num_points, dimension, inside=False, radius=radius)


# Generate random points that are well spaced in the [0,1] box.
def well_spaced_box(num_points, dimension):
    from numpy import random, ones, arange
    # Generate a "num_points^dimension" grid, making sure every grid
    # cell has at least one point in one of its dimensions.
    cell_width = 1 / num_points
    cells = ones((dimension, num_points)) * arange(num_points)
    # Randomly shuffle the selected grid cells for each point.
    for _ in map(random.shuffle, cells): pass
    # Convert the random selected cells into points.
    return cell_width * (random.random((dimension,num_points)) + cells).T


# Generate random points with a well spaced design over a unit ball.
def well_spaced_ball(num_points, dimension, inside=True):
    # Check the quality of the input.
    assert num_points >= 0, f"well_spaced_ball(num_points, dimension, inside=True), num_points >= 0 required but received num_points={num_points}"
    assert dimension >= 0,  f"well_spaced_ball(num_points, dimension, inside=True), dimension >= 0 required but received dimension={dimension}"
    # Import math functions for generating a well spaced design over a sphere.
    from numpy import pi, zeros, cos, sin, linspace, random, ones, arange, interp
    # Handle trivial case.
    if (min(num_points, dimension) == 0):
        return zeros((num_points, dimension))
    # Get well spaced random points over the unit cube in one less dimension
    #  as the source for spherical coordinates with a well spaced design.
    cell_width = 1 / num_points
    cells = ones((max(1,dimension-1), num_points)) * arange(num_points)
    # Randomly shuffle the selected grid cells for each point.
    for cell in cells: random.shuffle(cell)
    # Convert the random selected cells into points.
    coordinates = cell_width * (random.random((max(1,dimension-1), num_points)) + cells)
    # Exit early if only a single dimension is desired.
    if (dimension == 1): return 2*coordinates.T - 1
    # Push coordinates through an appropriate inverse density function to make
    #  the uniform density over the box into uniform density over the sphere.
    coordinates[-1,:] *= 2*pi    
    density_x = linspace(0, pi, 1000)
    density_gaps = density_x[1:] - density_x[:-1]
    density_y = ones(density_x.shape)
    for i in range(2, len(coordinates)+1):
        # For each coordinate, there is one addtional sin function.
        density_y *= sin(density_x)
        # Integrate a piecewise linear interpolant of the density function.
        density_cdf_y = zeros(density_y.shape)
        density_cdf_y[1:] += (
            density_gaps * (density_y[:-1] + (density_y[1:] - density_y[:-1]) / 2)
        ).cumsum()
        density_cdf_y[:] /= density_cdf_y[-1]
        # Interpolate the inverted CDF to transform a uniform random
        #  distribution into the desired distribution of data.
        coordinates[-i,:] = interp(
            coordinates[-i,:],
            density_cdf_y, density_x
        )
    # Convert the spherical coordinates into linear coordinates.
    points = zeros((dimension, num_points))
    points[0,:] = cos(coordinates[0,:])
    points[1:,:] = sin(coordinates[0,:])
    for i in range(1, dimension-1):
        points[i,:] *= cos(coordinates[i,:])
        points[i+1:,:] *= sin(coordinates[i,:])
    # Add random radii proportional to probability density
    #   on all contained spheres if points in the ball are requested.
    if (inside):
        radii = linspace(0, (num_points-1)/num_points, num_points)
        radii += random.uniform(size=(num_points,)) / num_points
        radii = radii ** (1 / dimension)
        random.shuffle(radii)
        points *= radii
    # Switch points to row vectors and return.
    return points.T


# Wrapper for well_spaced_ball routine.
def well_spaced_sphere(num_points, dimension):
    return well_spaced_ball(num_points, dimension, inside=False)


# This module provides some utilties involving random generation. The
# provided functions in this module are:
# 
#   random_ranage -- Generates a random sample from a range-like object.
#   cdf           -- Generates a random cumulative distribution function
#                    (designed to be used as a random variable).
#   ball          -- Generates random points inside N-dimension ball.
#   latin         -- Generates random points with latin cube design (well-spaced).
#   pairs         -- Generates random pairs of indices within a range.

# Custom excepion for improper user-specified ranges.
class InvalidRange(Exception):
    def __init__(self, start, stop, step, count):
        return super().__init__("\n\nError: random_range(start=%s,stop=%s,step=%s)[:count=%s] contains no elements."%(start,stop,step,count))

# Return a randomized "range" using the appropriate technique based on
# the size of the range being generated. If the memory footprint is
# small (<= 32K int's) then a random sample is created and returned.
# If the memory footprint would be large, a Linear Congruential
# Generator is used to efficiently generate the sequence.
# 
# Parameters are similar to the builtin `range` with:
#   start -- int, default of 0.
#   stop  -- int > start for no / positive step, < start otherwise.
#   step  -- int (!= 0), default of 1.
#   count -- int > 0, number of samples, default (stop-start)//step.
# 
# Usage (and implied parameter ordering):
#   random_range(a)             --> range(0, a, 1)[:a]
#   random_range(a, b) [a < b]  --> range(a, b, 1)[:b-a]
#   random_range(a, b, c)       --> range(a, b, c)[:(b-a)//c]
#   random_range(a, b, c, d)    --> range(a, b, c)[:min(d, (b-a)//c)]
#   random_range(a, d) [d <= a] --> range(0, a, 1)[:d]
# 
# If the size of the range is large, a Linear Congruential Generator is used.
#   Memory  -- O(1) storage for a few integers, regardless of parameters.
#   Compute -- O(n) at most 2 times the number of steps in the range, n.
# 
def random_range(start, stop=None, step=None, count=float('inf')):
    from random import sample, randint
    from math import ceil, log2
    # Add special usage where the second argument is meant to be a count.
    if (stop != None) and (stop <= start) and ((step == None) or (step >= 0)):
        start, stop, count = 0, start, stop
    # Set a default values the same way "range" does.
    if (stop == None): start, stop = 0, start
    if (step == None): step = 1
    # Compute the number of numbers in this range, update count accordingly.
    num_steps = int((stop - start) // step)
    count = min(count, num_steps)
    # Check for a usage error.
    if (num_steps == 0) or (count <= 0): raise(InvalidRange(start, stop, step, count))
    # Use robust random method if it has a small enough memory footprint.
    # Do not use this method if there is a number too large for C implementation.
    if (num_steps <= 2**15) and (max(map(abs,(start,stop,step))) < 2**63):
        for value in sample(range(start,stop,step), count): yield value
        return
    # Use the LCG for the cases where the above is too memory intensive.
    # Enforce that all numbers are python integers (no worry of overflow).
    start, stop, step = map(int, (start, stop, step))
    # Use a mapping to convert a standard range into the desired range.
    mapping = lambda i: (i*step) + start
    # Seed range with a random integer to start.
    value = int(randint(0,num_steps))
    # 
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    # 
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    # 
    offset = int(randint(0,num_steps)) * 2 + 1                 # Pick a random odd-valued offset.
    modulus = int(2**ceil(log2(num_steps)))                    # Pick a power-of-2 modulus just big enough to generate all numbers.
    multiplier = 4*(num_steps + int(randint(0,num_steps))) + 1 # Pick a multiplier 1 greater than a multiple of 4.
    if (modulus < 4): multiplier = 1
    # Track how many random numbers have been returned.
    found = 0
    while found < count:
        # If this is a valid value, yield it in generator fashion.
        if value < num_steps:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        value = (value*multiplier + offset) % modulus


# This function maps an index in the range [0, count*(count - 1) // 2] 
# to a tuple of integers in the range [0,count). The mapping will cover
# all pairs if you use all indices between [0, count*(count - 1) // 2].
def pair_to_index(p1, p2):
    if (p1 < p2): p1, p2 = p2, p1
    return (p1 * (p1 - 1) // 2) + p2


# This function maps an index in the range [0, count*(count - 1) // 2] 
# to a tuple of integers in the range [0,count). The mapping will cover
# all pairs if you use all indices between [0, count*(count - 1) // 2].
def index_to_pair(index):
    val = int(((1/4 + 2*index)**(1/2) + 1/2))
    remainder = index - val*(val - 1)//2
    return (val, remainder)


# Generate "count" random indices of pairs that are within "length" bounds.
# This only generates unique pairs with the first index always being
# greater than the second index provided (0-indexed).
def pairs(length, count=None):
    # Compute the hard maximum in the number of pairs.
    max_pairs = length*(length - 1) // 2
    # Initialize count if it wasn't provided.
    if type(count) == type(None): count = max_pairs
    count = min(count, max_pairs)
    # Get a random set of pairs (no repeats).
    for c,i in enumerate(random_range(count)):
        if (i >= count): break
        yield index_to_pair(i)
    print(" "*40, end="\r", flush=True)


# Generate a random uniform distribution of numbers deterministically.
def uniform(count, value=None, offset=4091, multiplier=1048573, max_numbers=2**32):
    max_numbers = max(count, max_numbers)
    # Seed the generate with random integers.
    if (value is None):
        value = max_numbers // 2
    if (offset is None):
        offset = int(np.random.randint(0,max_numbers))
    if (multiplier is None):
        multiplier = int(np.random.randint(0,max_numbers))
    # 
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    # 
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    # 
    offset = int(offset) * 2 + 1                  # Pick a random odd-valued offset.
    multiplier = 4*(max_numbers + multiplier) + 1 # Pick a multiplier 1 greater than a multiple of 4.
    modulus = int(2**np.ceil(np.log2(max_numbers)))     # Pick a modulus just big enough to generate all numbers (power of 2).
    # Return a fixed number of numbers.x
    for _ in range(count):
        yield value / modulus
        # Calculate the next value in the sequence.
        value = (value*multiplier + offset) % modulus

# Create a tensor mesh of "len(nodes)" in R^"dimension" space of all
# pairs of elements inside of "nodes". If "sample" is provided, then a
# random sample of size "sample" is drawn from that mesh.
# 
#    `mesh([0, 1], 2) = [[0, 0], [1, 0], [0, 1], [1, 1]]`
# 
def mesh(nodes, dimension, sample=None):
    # If a flat list of nodes is given, repeat it "dimension" times.
    if not hasattr(nodes[0], "__len__"):
        nodes = [nodes for i in range(dimension)]
    assert (len(nodes) == dimension), f"Provided list of nodes contained {len(nodes)} sublist{'s' if len(nodes) > 1 else ''}, but should either be flat or have one list per axis ({dimension} lists of numbers)."
    from numpy import asarray
    # Start of the actual function.
    from numpy import vstack, meshgrid
    if sample:
        # Compute the multiplicative product of a list of integers.
        # Make sure everything is PYTHON INTEGER to avoid overflow.
        def product(integers):
            p = int(integers[0])
            for i in range(1,len(integers)): p *= int(integers[i])
            return p
        mesh_size = product(list(map(len,nodes)))
        # Only continue if the sample does *not* include the whole grid.
        if (sample < mesh_size):
            from tlux.random import random_range
            from numpy import zeros
            # Compute the total mesh size (and track the size of each component).
            sizes = list(map(len, nodes))
            # Randomly draw points from the mesh.
            points = zeros((sample, dimension), dtype=float)
            for i,index in enumerate(random_range(mesh_size, count=sample)):
                smaller_prod = mesh_size
                # Decode the index into a set of indices for each component.
                for s in range(len(sizes)):
                    smaller_prod = int(smaller_prod // sizes[s])
                    curr_ind = int(index // smaller_prod )
                    # Handle the special case where remaining sizes are 1.
                    if (smaller_prod == 1):
                        curr_ind = min(curr_ind, sizes[s]-1)
                    # Subtract out the effect of the chosen index.
                    index -= curr_ind * smaller_prod
                    points[i,s] = nodes[s][curr_ind]
            # Return the randomly selected mesh points.
            return points
    # Default operation, return the full tensor product of "nodes".
    return vstack([_.flatten() for _ in meshgrid(*nodes)]).T
