"""Preview chunk selection utilities."""


def select_random(chunks, k: int):
    """Return ``k`` random chunk identifiers."""
    raise NotImplementedError


def select_diverse(chunks, k: int):
    """Return ``k`` diverse chunk identifiers using farthest point."""
    raise NotImplementedError
