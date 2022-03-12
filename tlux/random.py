
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

# Random points within a [0,1] box.
def box(num_points, dimension):
    from numpy.random import uniform
    return uniform(size=(num_points, dimension))

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
