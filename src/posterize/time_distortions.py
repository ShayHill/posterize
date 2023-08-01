"""Distortions that can be applie to time sequences.

input: time values from 0 to 1
output: time values from 0 to 1

:author: Shay Hill
:created: 2023-08-01
"""

import math
from typing import Iterable

Q1 = math.pi / 2
Q2 = Q1 * 2
Q3 = Q1 * 3


def interpolate_floats(
    times_a: Iterable[float], times_b: Iterable[float], time: float
) -> list[float]:
    """Interpolate between a and b for each a and b in times_a and times_b.

    :param times_a: The first set of times.
    :param times_b: The second set of times.
    :param time: The time to interpolate at. 0.0 returns times_a, 1.0 returns times_b.
    :return: The interpolated times.
    """
    interpolated = [a + time * (b - a) for a, b in zip(times_a, times_b)]
    # kludged in case of floating point errors
    interpolated = [min(1, max(0.0, t)) for t in interpolated]
    scale = 1 - 1 / len(interpolated)
    return [t * scale for t in interpolated]



def q1_time(times: Iterable[float], strength: float = 1) -> list[float]:
    """Distort time values with a sine wave in quadrant 1.

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    Samples crowd toward 1.
    """
    distorted = [math.sin(t * Q1) for t in times]
    return interpolate_floats(times, distorted, strength)


def q3_time(times: Iterable[float], strength: float = 1) -> list[float]:
    """Distort time values with a cos wave in quadrant 3.

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    Samples crowd toward 0.
    """
    distorted = [math.cos(Q2 + t * Q1) + 1 for t in times]
    return interpolate_floats(times, distorted, strength)


def cos_time(times: Iterable[float], strength: float = 1) -> list[float]:
    """Distort time values with a cosine wave over quadrants 3 and 4.

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    Samples crowd toward endpoints.
    """
    distorted = [(math.cos(Q2 + t * Q2) + 1) / 2 for t in times]
    return interpolate_floats(times, distorted, strength)


def sin_time(times: Iterable[float], strength: float = 1) -> list[float]:
    """Distort time values with a sine wave over quadrants 1 and 3.

    :param times: The times to distort. Float values (0.0, 1.0)
    :param strength: The strength of the distortion. 1.0 (default) is full distortion.
    :return: The distorted times. Float values (0.0, 1.0)

    Samples crowd toward 0.5.
    """
    distorted: list[float] = []
    for time in times:
        if time < 0.5:
            distorted.append(math.sin(time * Q2) / 2)
        else:
            distorted.append((math.sin(Q3 + (time - 0.5) * Q2)) / 2 + 1)
    return interpolate_floats(times, distorted, strength)


def pow_time(times: Iterable[float], power: float = 1) -> list[float]:
    """Distort times with a power function.

    :param times: The times to distort. Float values (0.0, 1.0)
    :param power: The strength of the distortion. 1.0 (default) is no distortion.
        Values > 1 crowd toward 1.0. Values < 1 crowd toward 0.0. Don't use negative
        numbers, because 0**neg will fail.
    :return: The distorted times. Float values (0.0, 1.0)

    Samples crowd toward 1.
    """
    distorted = [time**power for time in times]
    return interpolate_floats(times, distorted, 1)
