"""Distortions that can be applie to time sequences.

input: time values from 0 to 1
output: time values from 0 to 1

linear: return input
push_f: push first
push_l: push last
push_i: push in
push_o: push out

push_f and push_l give symmetrical results

    push_f => [0.00, 0.08, 0.29, 0.62, 1.00]
    push_l => [0.00, 0.38, 0.71, 0.92, 1.00]

    but they are not the inverse of each other.
    I.e., push_f(push_l(xs)) != xs && push_l(push_f(xs)) != xs

push_i and push_o give inverse results

    push_o => [0.00, 0.08, 0.29, 0.62, 1.00]
    push_i => [0.00, 0.25, 0.50, 0.75, 1.00]

    push_i(push_o(xs)) == xs && push_o(push_i(xs)) == xs

:author: Shay Hill
:created: 2023-08-01
"""

import itertools as it
from typing import Iterable, Sequence

import numpy as np


def _average_floats(
    floats_a: Sequence[float], floats_b: Sequence[float], time: float
) -> list[float]:
    """Interpolate between a and b for each a and b in times_a and times_b.

    :param floats_a: first sequence of values
    :param floats_b: second sequence of values
        (presumably the same number of items as floats_a)
    :param time: The time to interpolate at. 0.0 returns floats_a, 1.0 returns floats_b
    :return: The interpolated floats
    """
    min_val = min(it.chain(floats_a, floats_b))
    max_val = max(it.chain(floats_a, floats_b))
    interpolated = [a + (b - a) * time for a, b in zip(floats_a, floats_b)]
    return list(np.clip(interpolated, min_val, max_val))


def linear(times: Iterable[float]) -> list[float]:
    """This is a no-op. It returns the input."""
    return [t for t in times]


def push_f(times: Sequence[float], strength: float = 1) -> list[float]:
    """push_first, crowd values toward 0

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    push_f([0.00, 0.25, 0.50, 0.75, 1.00]) =>
           [0.00, 0.08, 0.29, 0.62, 1.00]

    Distort time values with a sin wave in quadrant 4.
    """
    q4 = np.interp(times, [0.0, 1.0], [np.pi * 1.5, np.pi * 2])
    distorted = np.interp([np.sin(r) for r in q4], [-1, 0], [0, 1])
    return _average_floats(times, list(distorted), strength)


def push_l(times: Sequence[float], strength: float = 1) -> list[float]:
    """push_last, crowd values toward 1

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    push_l([0.00, 0.25, 0.50, 0.75, 1.00]) =>
           [0.00, 0.38, 0.71, 0.92, 1.00]

    Distort time values with a sine wave in quadrant 1.
    """
    q1 = np.interp(times, [0.0, 1.0], [0, np.pi * 0.5])
    distorted = [np.sin(r) for r in q1]
    return _average_floats(times, distorted, strength)


def push_o(times: Sequence[float], strength: float = 1) -> list[float]:
    """push_out, crowd values away from 0.5

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    push_o([0.00, 0.25, 0.50, 0.75, 1.00]) =>
           [0.00, 0.15, 0.50, 0.85, 1.00]

    Distort time values with a cosine wave over quadrants 3 and 4.
    """
    q3q4 = np.interp(times, [0, 1], [np.pi, np.pi * 2])
    distorted = np.interp([np.cos(r) for r in q3q4], [-1, 1], [0, 1])
    return _average_floats(times, list(distorted), strength)


def push_i(times: Sequence[float], strength: float = 1) -> list[float]:
    """push_in, crowd values towards 0.5

    :param times: The distorted times to reverse.
    :param strength: The strength of the distortion. Should be the same as used in push_o function.
    :return: The original times before distortion.

    push_i([0.00, 0.25, 0.50, 0.75, 1.00]) =>
           [0.00, 0.33, 0.50, 0.67, 1.00]

    This is the inverse of push_o, so push_i(push_o(xs)) == xs
    """
    distorted = [np.arccos(x) / np.pi for x in np.interp(times, [0, 1], [1, -1])]
    return _average_floats(times, distorted, strength)
