<p align="center">
  <h1 align="center">tlux</h1>
</p>

<p align="center">
All the things necessary for reproducing work in my <a href="https://tchlux.github.io/research">research blog</a>.
</p>

## Includes
 - `tlux.approximate.axy`, a Fortran nonlinear regressor,
 - `tlux.approximate.balltree`, a Fortran ball tree nearest neighbor code,
 - `tlux.approximate.delaunay`, a Fortran simplicial interpolation code,
 - `tlux.unique`, a simplified and optimized C library for translating arrays of strings to same-shaped arrays of unique positive integers,
 - `tlux.regex`, a highly optimized and simplified C regular expression engine,
 - `tlux.data`, a pure Python Data class for row|column data storage,
 - `tlux.plot`, a simplified plotting package wrapped around Plotly,
 - `tlux.math`, convenient python 'polynomial', 'fraction', and 'spline' objects as well as 'SVD' and 'regular_simplex' functions,
 - `tlux.random`, a collection of miscellaneous random value generation functions,
 - `tlux.profiling`, an easy-to-use python profiling decorator,


## INSTALLATION:

  Install the latest stable release with:

```bash
python3 -m pip install --user tlux
```

  In order to install the current files in this repository
  (potentially less stable) use:

```bash
python3 -m pip install --user git+https://github.com/tchlux/tlux.git
```

## USAGE:

### Python

```python
import tlux
help(tlux)
```

  Descriptions of the contents might follow.

### Command line

```bash
python -m tlux [--clean] [--build] [-h] [--help]
```

  Run the `tlux` package.
