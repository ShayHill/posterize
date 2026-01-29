"""Cluster 1d values.

:author: Shay Hill
:created: 2026-01-27
"""

import itertools as it
from collections.abc import Iterable
from functools import partial

import numpy as np
from numpy import typing as npt
from typing import Annotated, TypeAlias

from basic_colormath import float_to_8bit_int

_Indices: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r, c)"]
_Palette: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n, 3)"]


def _aggregate_and_sort_floats(
    values: Iterable[float], weights: Iterable[float]
) -> tuple[list[float], list[float]]:
    """Return unique values and their aggregated weights, sorted by value."""
    nos = list(values)
    wts = list(it.islice(it.cycle(weights), len(nos)))
    value_to_weight: dict[float, float] = {}
    for val, weight in zip(nos, wts, strict=True):
        value_to_weight[val] = value_to_weight.get(val, 0.0) + weight
    sorted_pairs = sorted(value_to_weight.items(), key=lambda x: x[0])
    nos = [n for n, _ in sorted_pairs]
    wts = [w for _, w in sorted_pairs]
    return (nos, wts)


def _get_segment_cost(
    prefix_sum: list[float],
    prefix_sqd: list[float],
    prefix_wts: list[float],
    left: int,
    right: int,
) -> float:
    """Compute weighted variance cost for segment [left, right]."""
    total_weight = prefix_wts[right + 1] - prefix_wts[left]
    if total_weight <= 0:
        return 0.0
    weighted_sum = prefix_sum[right + 1] - prefix_sum[left]
    weighted_sq_sum = prefix_sqd[right + 1] - prefix_sqd[left]
    return weighted_sq_sum - weighted_sum * weighted_sum / total_weight


def cluster_floats(
    numbers: Iterable[float], k: int, weights: Iterable[float] = (1,)
) -> tuple[list[float], list[list[float]]]:
    """Return optimal cluster centers and their members for 1D data.

    :param values: Iterable of values to cluster
    :param k: Number of clusters to create
    :param weights: Optional iterable of weights for each value
    :return: tuple of (exemplars, cluster_members) where exemplars is a list
        of floats (the cluster centers) and cluster_members is a list of
        lists, each containing the values in that cluster

    Uses the Fisher-Jenks natural breaks (optimal 1d k-means) algorithm.
    """
    if k < 1 or not numbers:
        return ([], [])
    nos, wts = _aggregate_and_sort_floats(numbers, weights)
    n = len(nos)
    k = min(k, n)

    prefix_sum = [0.0, *it.accumulate(n * w for n, w in zip(nos, wts, strict=True))]
    prefix_sqd = [0.0, *it.accumulate(n * n * w for n, w in zip(nos, wts, strict=True))]
    prefix_wts = [0.0, *it.accumulate(wts)]

    segment_cost = partial(_get_segment_cost, prefix_sum, prefix_sqd, prefix_wts)

    dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    prev = [[-1] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            for p in range(j - 1, i):
                cost = dp[p][j - 1] + segment_cost(p, i - 1)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = p
    exemplars: list[float] = []
    cluster_members: list[list[float]] = []
    i = n
    for j in reversed(range(1, k + 1)):
        p = prev[i][j]
        total_weight = prefix_wts[i] - prefix_wts[p]
        if total_weight > 0:
            weighted_sum = prefix_sum[i] - prefix_sum[p]
            mean = weighted_sum / total_weight
            exemplars.append(mean)
            cluster_members.append(nos[p:i])
        i = p

    return (exemplars, cluster_members)


def cluster_uint8(
    numbers: Iterable[int], k: int, weights: Iterable[float] = (1,)
) -> tuple[list[int], list[list[int]]]:
    """Return optimal cluster centers and their members for 8-bit integer values.

    Assumes all values are in [0, 255].
    """
    exemplars, members = cluster_floats(numbers, k, weights)
    exemplars_ = [float_to_8bit_int(x) for x in exemplars]
    members_ = [[int(i) for i in m] for m in members]
    return exemplars_, members_


def quantize_mono_pixels(pixels: _Indices, k: int) -> tuple[_Palette, _Indices]:
    """Quantize monochrome pixels to a palette and indices."""
    exemplars, members = cluster_uint8(pixels.flatten(), k)
    return palette, indices


# ────────────────────────────────────────────────
#                   Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    wwww = [1, 1, 1, 2, 3]

    exemplars, members = cluster_floats(data, 3, wwww)
    print("Exemplars:", exemplars)  # noqa: T201
    print("Members:", members)  # noqa: T201

    data = [1, 2, 3, 4, 4, 5, 5, 5]
    data = [0] + [1] * 100 + [2]
    exemplars, members = cluster_uint8(data, 2)
    print("Exemplars:", exemplars)  # noqa: T201
    print("Members:", members)  # noqa: T201
