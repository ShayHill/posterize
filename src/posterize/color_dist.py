"""A contrast scheme to identify and remove overlap between colors.

overlaps

This imagines each color in the rgb cube as a bottle of transparent ink into which
pigments red, green, blue, white, cyan, magenta, yellow, and black are poured. The
pigments are mixed. Only the ratio of the pigments matters, not the total volume,
because the result is always opaque. So, pink would be a bottle of red and white
pigments. If you remove red, it gets more white; if you remove white, it gets more
red; if you remove blue, nothing happens. If you remove all of the pigment, you get a
value error, because there is no alpha channel in this model.

The pink example would almost work with the rgb model, but this 8-channed model
treats white and red differently. If you remove red from white, you still get white,
not the cyan you would get with an additive color model. You could also remove black
from gray and get white. This model is not additive or subtractive, but a mix of
both.

:author: Shay Hill
:created: 2024-04-19
"""


from __future__ import annotations
import dataclasses
import itertools as it
from operator import attrgetter
from typing import Annotated, Iterable, cast

from numpy import typing as npt
import numpy as np

from basic_colormath import float_to_8bit_int

Rgb = Annotated[tuple[int, int, int], "3 ints [0, 255]"]
Rgbwcmyk = Annotated[tuple[int, int, int, int, int, int, int, int], "8 ints [0, 255]"]

Floats8Arg = Annotated[Iterable[float], "8 floats"]
Floats8Out = tuple[float, float, float, float, float, float, float, float]


def _clamp(value: float) -> float:
    """Clamp a float to the range 0-255."""
    return max(0, min(value, 255))


def _clamp8(eight_floats: Floats8Arg) -> Floats8Out:
    """Clamp an octuple to the range 0-255."""
    r, g, b, w, c, m, y, k = (_clamp(a) for a in eight_floats)
    return r, g, b, w, c, m, y, k


def _scale8(eight_floats: Floats8Arg, scale: float) -> Floats8Out:
    """Scale an octuple by a float, clamp to [0, 255]."""
    r, g, b, w, c, m, y, k = (_clamp(a * scale) for a in eight_floats)
    return r, g, b, w, c, m, y, k


def _sub8(minuend: Floats8Arg, subtrahend: Floats8Arg) -> Floats8Out:
    """Subtract two octuple values."""
    r, g, b, w, c, m, y, k = cast(
        Floats8Arg, (a - b for a, b in zip(minuend, subtrahend))
    )
    return r, g, b, w, c, m, y, k


def _normalize8(eight_floats: Floats8Arg) -> Rgbwcmyk:
    """Normalize an octuple of floats to 8-bit integers.

    :param floats8: tuple of 8 floats
    :return: tuple of 8 integers, summing to 255
    :raises ValueError: if no color information is present (all zero floats)

    Scale octuple to integers that scale to 255.

    The incremental increases and decreases in the `while sum(as_ints)` blocks will
    most likely never be required. They are there to handle rare rounding errors when
    "rounding" floats to 8-bit integers.
    """
    clamped = _clamp8(eight_floats)

    if not sum(clamped):
        msg = f"Cannot normalize {clamped}. Does not contain color information"
        raise ValueError(msg)

    scaled = _scale8(clamped, 255 / sum(clamped))
    as_ints = [float_to_8bit_int(f) for f in scaled]

    while sum(as_ints) > 255:
        as_ints[as_ints.index(max(as_ints))] -= 1
    while sum(as_ints) < 255:
        as_ints[as_ints.index(min(as_ints))] += 1

    r, g, b, w, c, m, y, k = as_ints
    return r, g, b, w, c, m, y, k


@dataclasses.dataclass
class ColorDist:
    """Define an rgb color as a distribution of 8 values.

    rgbwcmyk: tuple[int, int, int, int, int, int, int, int]
    """

    dist: tuple[int, int, int, int, int, int, int, int]

    @property
    def rgb(self) -> Rgb:
        """Get the color dist as an rgb tuple ([0..255], [0..255], [0..255])."""
        r = self.dist[0] + self.dist[3] + self.dist[5] + self.dist[6]
        g = self.dist[1] + self.dist[3] + self.dist[4] + self.dist[6]
        b = self.dist[2] + self.dist[3] + self.dist[4] + self.dist[5]
        return (r, g, b)

    @classmethod
    def from_rgb(cls, rgb: Rgb) -> ColorDist:
        """Set the distribution from an rgb tuple."""
        r, g, b = rgb
        w = min(r, g, b)
        k = 255 - max(r, g, b)
        y = min(r, g) - w
        c = min(g, b) - w
        m = min(b, r) - w
        r -= max(m, y) + w
        g -= max(y, c) + w
        b -= max(c, m) + w
        return cls(dist=(r, g, b, w, c, m, y, k))

    @classmethod
    def intersection(cls, *dists: ColorDist) -> Floats8Out:
        """Get the commonality of multiple distributions."""
        channels = zip(*map(attrgetter("dist"), dists))
        r, g, b, w, c, m, y, k = (min(c) for c in channels)
        return (r, g, b, w, c, m, y, k)

    @classmethod
    def intersection_volume(cls, *dists: ColorDist) -> float:
        """Measure the volume of the intersection of multiple distributions."""
        intersection = cls.intersection(*dists)
        return sum(intersection)

    @classmethod
    def from_intersection(cls, *dists: ColorDist) -> ColorDist:
        """Create a distribution from the intersection of multiple distributions."""
        intersection = cls.intersection(*dists)
        if sum(intersection) == 0:
            msg = f"No intersection found in {dists}"
            raise ValueError(msg)
        return cls(dist=_normalize8(intersection))

    def take(self, other: ColorDist, amount: float) -> ColorDist:
        """Take a portion of another distribution."""
        subtrahend = _scale8(other.dist, amount)
        difference = _sub8(self.dist, subtrahend)
        return self.__class__(dist=_normalize8(difference))

    def take_min(self, other: ColorDist) -> ColorDist:
        """Take the smallest amount of other that changes self.rgb.

        If self and other are the same, return a copy of self. E.g., pure red minus
        pure red is still red, because no other pigment exists.
        """
        if self.dist == other.dist:
            return self.__class__(dist=self.dist)

        input_rgb = self.rgb
        for amount in (x / 255 for x in range(1, 256)):
            result = self.take(other, amount)
            if result.rgb != input_rgb:
                return result

        return self.__class__(dist=self.dist)


def _remove_overlap_step(rgbs: Iterable[Rgb]) -> list[Rgb]:
    """Remove overlap from a list of rgb values.

    Given a group of rgb colors, remove a small amount of the common color from each.
    If no intersection exists, remove nothing.
    """
    dists = [ColorDist.from_rgb(rgb) for rgb in rgbs]
    try:
        intersection = ColorDist.from_intersection(*dists)
    except ValueError:
        # no intersection
        return list(rgbs)
    return [dist.take_min(intersection).rgb for dist in dists]


def measure_overlap(rgbs: Iterable[Rgb]) -> float:
    """Measure the overlap in a list of rgb values."""
    return ColorDist.intersection_volume(*map(ColorDist.from_rgb, rgbs))


def measure_max_overlap(rgbs: Iterable[Rgb]) -> float:
    """Measure the maximum overlap in a list of rgb values."""
    pairs = it.combinations(map(ColorDist.from_rgb, rgbs), 2)
    return max(ColorDist.intersection_volume(*pair) for pair in pairs)


def _add_contrast_step(rgbs: Iterable[Rgb], max_overlap: float) -> list[Rgb]:
    """Add contrast to a list of rgb values."""
    # start with most overlapping pair
    rgbs = list(rgbs)
    pairs = it.combinations(rgbs, 2)
    overlapping_subset = max(pairs, key=measure_overlap)
    if measure_overlap(overlapping_subset) <= max_overlap:
        return rgbs

    # try to grow the overlapping set
    remaining = [x for x in rgbs if x not in overlapping_subset]
    while remaining:
        candidates = (overlapping_subset + (x,) for x in remaining)
        next_candidate = max(candidates, key=measure_overlap)
        if measure_overlap(next_candidate) <= max_overlap:
            break
        overlapping_subset = next_candidate
        remaining = [x for x in rgbs if x not in overlapping_subset]

    for old, new in zip(overlapping_subset, _remove_overlap_step(overlapping_subset)):
        rgbs[rgbs.index(old)] = new

    return rgbs


def subtract_overlap_to_improve_contrast(
    rgbs: Iterable[Rgb], max_overlap: float
) -> list[Rgb]:
    """Add contrast to a list of rgb values."""
    rgbs = list(rgbs)
    seen = {tuple(rgbs)}

    while True:
        new_rgbs = _add_contrast_step(rgbs, max_overlap)
        if tuple(new_rgbs) in seen:
            return rgbs
        seen.add(tuple(new_rgbs))
        rgbs = new_rgbs



def desaturate_by_color(
    pixels: npt.NDArray[np.float64], color: Rgb
) -> npt.NDArray[np.float64]:
    """Desaturate an image by a color."""
    ww, hh = pixels.shape[:2]
    dist = ColorDist.from_rgb(color)
    pixels_dists = np.array([ColorDist.from_rgb(p) for p in pixels.reshape(-1, 3)])
    return np.array([ColorDist.intersection_volume(dist, p) for p in pixels_dists]).reshape(ww, hh)
