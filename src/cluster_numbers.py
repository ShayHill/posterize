"""Cluster 1d values.

:author: Shay Hill
:created: 2026-01-27
"""

from collections.abc import Iterable
import itertools as it
from basic_colormath import float_to_8bit_int


def get_optimal_exemplars(
    values: Iterable[int],
    k: int,
    weights: Iterable[float] | None = None,
) -> list[float]:
    """Return a list of k optimal cluster centers for 1D data.

    Uses dynamic programming. Values assumed to be in (0, 255),
    len(values) ≤ 255. Result is sorted by cluster position.

    :param values: Iterable of integer values to cluster
    :param k: Number of clusters to create
    :param weights: Optional iterable of weights for each value
    :return: list of floats (the exemplars / representatives)
    """
    if k < 1 or not values:
        return []

    nos = list(values)
    if weights is None:
        weights = [1.0]
    wts = list(it.islice(it.cycle(weights), len(nos)))

    # Aggregate duplicate values by summing their weights
    value_to_weight: dict[int, float] = {}
    for val, weight in zip(nos, wts, strict=True):
        value_to_weight[val] = value_to_weight.get(val, 0.0) + weight
    sorted_pairs = sorted(value_to_weight.items(), key=lambda x: x[0])
    nos = [pair[0] for pair in sorted_pairs]
    wts = [pair[1] for pair in sorted_pairs]

    n = len(nos)
    k = min(k, n)

    # Weighted prefix sums for O(1) range statistics
    prefix_sum = [0.0] * (n + 1)
    prefix_sqd = [0.0] * (n + 1)
    prefix_wts = [0.0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + nos[i] * wts[i]
        prefix_sqd[i + 1] = prefix_sqd[i] + nos[i] * nos[i] * wts[i]
        prefix_wts[i + 1] = prefix_wts[i] + wts[i]

    def segment_cost(left: int, right: int) -> float:
        """Compute weighted variance cost for segment [left, right]."""
        total_weight = prefix_wts[right + 1] - prefix_wts[left]
        if total_weight <= 0:
            return 0.0
        weighted_sum = prefix_sum[right + 1] - prefix_sum[left]
        weighted_sq_sum = prefix_sqd[right + 1] - prefix_sqd[left]
        return weighted_sq_sum - weighted_sum * weighted_sum / total_weight

    INF = 1e100
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
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
    i = n
    j = k
    while j > 0:
        p = prev[i][j]
        total_weight = prefix_wts[i] - prefix_wts[p]
        if total_weight > 0:
            weighted_sum = prefix_sum[i] - prefix_sum[p]
            mean = weighted_sum / total_weight
            exemplars.append(mean)
        i = p
        j -= 1

    return [float_to_8bit_int(x) for x in exemplars]


# ────────────────────────────────────────────────
#                   Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    wwww = [1, 1, 1, 2, 3]

    exemplars = get_optimal_exemplars(data, 3, wwww)
    print("Exemplars:", [x for x in exemplars])  # noqa: T201
    # # → Exemplars: [1.4, 49.267, 221.475]

    data = [1, 2, 3, 4, 4, 5, 5, 5]
    exemplars = get_optimal_exemplars(data, 3)
    print("Exemplars:", [x for x in exemplars])  # noqa: T201

    # exemplars4 = get_optimal_exemplars(data, 4)
    # print("With 4 clusters:", [round(x, 3) for x in exemplars4])  # noqa: T201
    # # → With 4 clusters: [1.4, 49.267, 210.55, 237.4]
