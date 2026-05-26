# AXY Project Notes

This directory is a personal research project toward a learner: a general
digital intelligence built from simple constituent parts. Prefer small,
verifiable changes. The most important current work is hardening `axy.f90` and
its Python harness on small problems whose behavior can be observed exactly.

## Working Style

- Simplicity is the default. Make the fewest modifications and add the fewest
  lines that solve the observed problem.
- Do not preserve obsolete compatibility unless explicitly asked. Prefer
  deletion and simplification when code is clearly stale.
- When diagnosing bugs, first make the problem visible and repeatable. Trace the
  cause before patching.
- `AXY.fit` is intentionally memory-conservative. It is expected to mutate
  inputs and allocate as little as possible.
- Tests here are direct Python files, not pytest. Use the existing `_test_*`
  functions in `test/test_axy.py`.

## Architecture

- `axy.f90` is the authoritative compiled implementation. It contains model
  evaluation, embedding, aggregation, gradient calculation, fitting, and support
  routines.
- `axy_py.py` is the local Python reference harness. Its `evaluate` function is
  intended to match Fortran `EVALUATE` closely enough to serve as an independent
  oracle for small cases.
- `test/test_axy.py` is the main self-authored test harness. Important entry
  points include `_test_fetch_data`, `_test_evaluate`, and
  `_test_model_gradient`.
- `test/scenarios.py` builds scenario/config/data combinations. Use it rather
  than manually inventing broad architecture matrices.
- `summary.AxyModel` is useful for inspecting packed models and gradient blocks.

## Model Concepts

- AXY has an aggregate model and an optional fixed model. The aggregate model
  can feed the fixed model or produce final outputs directly.
- Aggregate data is grouped by `sizes`. Aggregate starts and fixed starts are
  derived by `COMPUTE_BATCHES` in Fortran.
- The aggregate model outputs `ADO` columns. These are aggregate values that are
  averaged across each aggregate group.
- `PAIRWISE_AGGREGATION` expands aggregate inputs into pair comparisons.
- `PARTIAL_AGGREGATION` emits suffix/running aggregate outputs into the fixed
  model instead of one output per aggregate group.
- Categorical inputs are embedded by `EMBED` / `UNPACK_EMBEDDINGS`; pairwise
  categorical aggregate inputs encode embedding differences.

## Gradient Understanding

- `MODEL_GRADIENT` computes the local gradient for the packed model variables.
  The core backward path is `BASIS_GRADIENT`, with aggregation handled by
  `COMPUTE_AGGREGATION_GRADIENT` and basis backprop by
  `UNPACKED_BASIS_GRADIENT`.
- For non-partial aggregation, every aggregate row in a group receives the same
  output gradient divided by the group size.
- For partial aggregation, each aggregate row contributes to all suffix averages
  that include it. The direct final partial output is the raw final aggregate
  value.
- Aggregate backprop calls `UNPACKED_BASIS_GRADIENT` with `EXTRA=0`; there is no
  extra aggregate weight column.

## Gradient Oracle

- `_test_model_gradient` uses the Python forward pass as the truth source.
- The oracle uses exact rational arithmetic through `TracedFraction`, a
  test-local wrapper around `tlux.math.fraction.Fraction`.
- `TracedFraction` records comparison outcomes during Python forward evaluation.
  A finite-difference sample is accepted only when its comparison trace matches
  the base evaluation trace.
- The finite-difference harness tries `+step`, then `-step`, then repeatedly
  halves the step. Coordinates that cannot find a same-path perturbation are
  treated as boundary-ambiguous rather than gradient failures.
- This makes the oracle strong for gradients inside a fixed piecewise-linear
  region and prevents false failures at activation or clamp boundaries.

## Current Confidence And Gaps

- Numeric-output gradients are currently verified across the scenario generator
  for fixed-only, aggregate-only, aggregate-to-fixed, linear/layered, partial,
  pairwise, weighted-output, and threaded cases exercised by
  `_test_model_gradient`.
- Categorical output gradient coverage is still incomplete; `_test_model_gradient`
  skips categorical output scenarios.
- The fit loop is not proven by the gradient tests. Treat optimizer behavior,
  conditioning, and long-run fitting dynamics as separate concerns.
- Exact boundary subgradient conventions are not fully proven; the test harness
  avoids treating boundary ambiguity as a gradient failure.
