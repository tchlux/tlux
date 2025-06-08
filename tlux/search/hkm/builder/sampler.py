"""Sampling utilities for HKM builder."""

from typing import Iterable


def reservoir_sample(iterable: Iterable, k: int):
    """Return ``k`` items uniformly sampled from ``iterable``."""
    raise NotImplementedError


def kmeanspp_sample(data, k: int):
    """Return ``k`` initial centroids using the k-means++ method."""
    raise NotImplementedError
