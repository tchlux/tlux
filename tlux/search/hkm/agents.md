# Coding Guidelines

## Core Principles
- **Lean & self-contained** – standard library + NumPy only is preferred; add dependencies only with a material and quantified benefit.
- **Pedagogical clarity** – names and comments teach; code is decipherable without a debugger.
- **Deterministic efficiency** – predictable memory/CPU, seeded RNGs, streaming where possible.
- **Maintainability first** – optimize after profiling, not before.
- **ASCII-only source** – no Unicode or special characters inside code or comments.

## Layout & Naming
- **Banner (triple-quoted)** – 1-sentence synopsis,  <= 30-line overview, example usage.
- **Imports** – stdlib -> third-party -> intra-package, one per line, no wildcards.
- **Constants / private helpers** – `_snake_case`, `UPPER_SNAKE` for constants.
- **Typing** – functions use full type hints on all inputs and output.
- **Naming** – `snake_case.py`, `PascalCase`, `snake_case`, `_private`, `UPPER_SNAKE`, CLI flags like `--max-iter`.
- **CLI & demo** – under `if __name__ == "__main__":`, <= 25 LOC, zero side-effects on import.

## Documentation & Comments
- **Module banner** as above.
- **API docs** – NumPy-style comment blocks with "#" *preceding* classes / functions  
  (`# Description:\n#  ...\n# \n# Parameters:\n#   arg (type): description\n# ...`), no inline `"""docstrings"""` inside definitions they are ugly do NOT use docstrings.
- Inline comments provide "overview descriptions" for intents of blocks of code and fill explanations where reading the code alone is not sufficiently obvious to indicate why something is being done.
- Avoid horizontal lines (e.g., `---(-)*` or `###(#)*`), rather rely on simple whitespace and vertical separation.

## API, Types & Error Contracts
- PEP 484 everywhere public; prefer concrete dtypes (`np.float32`, `np.int32`) and note shapes `(n, d)`.
- Validate inputs early, raise `ValueError`, `TypeError`, or `RuntimeError` with explicit messages.
- Use the message to state the violated contract (“k must be in [1, n]”).

## Performance & I/O
- Use seeded random generators with default values.
- Minimize memory footprint where possible, vectorize with NumPy; memory-map large arrays.
- O(1) or single-pass algorithms are preferred; profile before micro-optimizing.
- Avoid `print` unless in demonstration code; internal error logs use `logging`.

## Testing & Examples
- Each algorithm has a doctest-style example in the preceding comments.
- Unit tests reside in local `./tests/` subdirectories and not inside source files.
- Every file should be runnable as a quick sanity check via its `__main__` if it does not already have a command line to support.
- Activate the local virtual environment when testing with commands like `source .env/bin/activate && python3 -m pytest tests/` to execute all tests.
- No test should take more than 30 seconds, so you are encouraged to set timeouts on test execution to enforce that behavior.


## Example Python Code

```
# Create a directory at the specified path, return 
# a boolean indicating if that directory already existed.
#
# Parameters:
#   path (str): Directory path.
#   exist_ok (bool): Allow existing directory without error.
#
# Returns:
#   (bool): True 
# 
# Raises:
#   OSError: If creation fails.
# 
def mkdir(self, path: str, exist_ok: bool = False) -> bool:
    # Check existence of the directory, accepting the slight chance of race condition.
    existed: bool = os.path.exists(path)
    # Make the directory.
    os.makedirs(path, exist_ok=exist_ok)
    return existed
```

Be mindful of the goals of creating a MINIMAL and CLEAN library. Redundancy should be stripped. Every line of code should be added only with clear and necessary purpose. Constantly try to discover new ways to refactor and reduce unique logic and delete significant chunks of code to simplify and compartmentalize the logic often.

If any pattern is repeated more than twice, it should be abstracted out and compartmentalized for reuse to minimize total unique logic in the code base.
