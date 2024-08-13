"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""

from __future__ import annotations
import itertools as it
import logging
from pathlib import Path
import dataclasses

from typing import Iterator
import numpy as np
import numpy.typing as npt
from basic_colormath import (
    float_to_8bit_int,
    float_tuple_to_8bit_int_tuple,
    get_delta_e,
)
from cluster_colors import KMedSupercluster, get_image_clusters
from cluster_colors.clusters import Member
from cluster_colors.cut_colors import cut_colors
from cluster_colors.pool_colors import pool_colors
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from PIL import Image
from svg_ultralight import new_element, update_element, write_root
from svg_ultralight.strings import svg_color_tuple

from posterize.color_dist import desaturate_by_color
from posterize.image_arrays import (
    _write_bitmap_from_array,
    get_image_pixels,
    write_monochrome_bmp,
)
from posterize.paths import PROJECT, TEMP, WORKING
from posterize.svg_layers import SvgLayers

logging.basicConfig(level=logging.INFO)

_MAX_8BIT = 255
_BIG_INT = 2**32 - 1
_BIG_SCALE = _BIG_INT / _MAX_8BIT


def get_vivid(color: tuple[float, float, float]) -> float:
    """Get the vividness of a color.

    :param color: color to get the vividness of
    :return: the vividness of the color
    """
    r, g, b = color
    return max(r, g, b) - min(r, g, b)


def get_clusters(colors: npt.NDArray[np.float64]) -> KMedSupercluster:
    """Pool and cut colors.

    :param colors: colors to pool and cut
    :return: pooled and cut colors
    """
    pooled = pool_colors(colors)
    pooled_and_cut = cut_colors(pooled, 512)
    members = Member.new_members(pooled_and_cut)
    return KMedSupercluster(members)


_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")


SCREEN_HEAD = (
    r'<svg xmlns="http://www.w3.org/2000/svg" '
    r'xmlns:xlink="http://www.w3.org/1999/xlink" '
    r'viewBox="0 0 {0} {1}" width="{0}" height="{1}">'
)
SCREEN_TAIL = b"</svg>"

# def _get_rgb_pixels(path: Path) -> npt.NDArray[np.uint8]:
#     """Get the RGB pixels of an image.

#     :param path: path to the image
#     :return: RGB pixels of the image
#     """
#     image = Image.open(path)
#     return np.array(image)


def _get_iqr_bounds(floats: npt.NDArray[np.float64]) -> tuple[np.float64, np.float64]:
    """Get the IQR bounds of a set of floats.

    :param floats: floats to get the IQR bounds of
    :return: the IQR bounds of the floats (lower, upper)

    Use the interquartile range to determine the bounds of a set of floats (ignoring
    outliers).
    """
    q25 = np.quantile(floats, 0.25)
    q75 = np.quantile(floats, 0.75)
    iqr = q75 - q25
    lower: np.float64 = max(q25 - 1.5 * iqr, np.min(floats))
    upper: np.float64 = min(q75 + 1.5 * iqr, np.max(floats))
    return lower, upper


def _normalize_errors_to_8bit(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Normalize a grid of floats (error delta per pixel) to 8-bit integers.

    :param grid: grid to normalize - array[float64] shape=(h, w)
    :return: normalized grid - array[uint8] shape=(h, w)

    Here, this takes a grid of floats representing the difference in error-per-pixel
    between two images. Where

    * errors_a = the error-per-pixel between the target image and the current state
    * errors_b = the error-per-pixel between the target image some potential state

    The `grid` arg here is errors_b - errors_a, so the lowest (presumably negative)
    values are where the potential state is better than the current state, and the
    highest values (presumably postive) are where the potential state is worse.

    The return value is a grid of the same shape with the values normalized to 8-bit
    integers, clipping outliers. This output grid will be used to create a monochrome
    bitmap where the color of each pixel represents the improvement we would see if
    using the potential state instead of the current state on that pixel.

    * 0: potential state is better
    * 255: potential state is worse
    """
    lower, upper = _get_iqr_bounds(grid.flatten())
    shift = 0 - lower
    scale = 255 / (upper - lower)
    grid = np.clip((grid + shift) * scale, 0, 255)
    return ((grid * _BIG_SCALE).astype(np.uint32) >> 24).astype(np.uint8)


class TargetImage:
    """A type to store input images and current error."""

    def __init__(self, path: Path, elements: list[EtreeElement] | None = None) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param lux: number of quantiles to use in error calculation
        """
        self.path = path
        self.elements: list[EtreeElement] = elements or []

        image = Image.open(path)
        self._bhead = SCREEN_HEAD.format(image.width, image.height).encode()

        self.clusters = get_image_clusters(path)
        bg_color = self.get_colors(1)[0]
        self.set_background(bg_color)

        self._image_grid = np.array(image).astype(float)
        self._state_grid = self.get_solid_grid(bg_color)
        self._error_grid = self.get_error(self._state_grid)

    @property
    def sum_error(self) -> float:
        return float(np.sum(self._error_grid))

    def get_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the most common colors in the image, largest to smallest
        """
        self.clusters.split_to_at_most(num)
        colors: list[tuple[int, int, int]] = []
        for r, g, b in self.clusters.get_rsorted_exemplars():
            r = min(255, max(0, r))
            g = min(255, max(0, g))
            b = min(255, max(0, b))
            colors.append(float_tuple_to_8bit_int_tuple((r, g, b)))
        return colors

    def get_next_color(self, idx: int = 0) -> tuple[float, float, float]:
        """Get the next color to use.

        :return: the next color to use
        """
        if not self.elements:
            self.clusters.split_to_at_most(1)
            r, g, b = self.clusters.get_rsorted_exemplars()[0]
            return r, g, b
        self.clusters.split_to_at_most(8)
        r, g, b = self.clusters.get_rsorted_exemplars()[0]
        return r, g, b

    def set_background(self, color: tuple[int, int, int]) -> None:
        """Replace the background with the current background color."""
        bg_element = new_element(
            "rect", width="100%", height="100%", fill=f"rgb{color}"
        )
        self.elements[:1] = [bg_element]

    def get_solid_grid(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Get a solid color grid the same shape as self._image_grid.

        :param color: color to make the grid
        :return: solid color grid
        """
        return np.full_like(self._image_grid, color)

    def get_error(self, grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Get the error-per-pixel between a pixel grid and self._image_grid.

        :param grid: array[n,m,3] grid to compare to self._image_grid
        :return: error-per-pixel [n,m,1] between grid and self._image_grid
        """
        return np.sum((self._image_grid - grid) ** 2, axis=2) ** (1 / 2)

    def _get_candidate_error_grid(
        self, candidate: EtreeElement | None = None
    ) -> npt.NDArray[np.float64]:
        if candidate is None:
            state_grid = self._state_grid
        else:
            state_grid = self._raster_state(candidate, WORKING / "candidate.png")
        return self.get_error(state_grid)

    def get_candidate_error(self, candidate: EtreeElement | None = None) -> float:
        return float(np.sum(self._get_candidate_error_grid(candidate)))

    def get_last_used_color(self) -> tuple[int, int, int]:
        """Get the last color used.

        :return: the color of the last element added
        """
        rgb_str = self.elements[-1].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = (int(x) for x in rgb_str.split(","))
        return (r, g, b)

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append
        """
        self.elements.append(element)
        self._state_grid = self._raster_state(element, WORKING / "state.png")
        self._error_grid = self._get_candidate_error_grid()

        ws = self._error_grid[..., np.newaxis]
        needs = np.concatenate([self._image_grid, ws], axis=2).reshape(-1, 4)
        self.clusters = get_clusters(needs)

    def _raster_state(
        self, candidate: EtreeElement, filename: Path
    ) -> npt.NDArray[np.float64]:
        elements = [*self.elements, candidate]
        bstrings = map(etree.tostring, elements)
        root = etree.fromstring(
            self._bhead + b"\n" + b"\n".join(bstrings) + SCREEN_TAIL
        )
        _ = write_root(_INKSCAPE, filename, root, do_png=True)
        image = Image.open(filename)
        return np.array(image).astype(float)[:, :, :3]

    def get_color_error_delta(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """What is the difference in error between the current state and a solid color?"""
        solid_color = np.full_like(self._image_grid, color)
        color_error = self.get_error(solid_color)
        error_delta = color_error - self._error_grid
        return _normalize_errors_to_8bit(error_delta)

    def evaluate_next_color(
        self, idx: int = 0
    ) -> tuple[tuple[float, float, float], npt.NDArray[np.uint8]]:
        """Evaluate the next color.

        :return: the error of the next color
        """
        next_color = self.get_next_color(idx)
        return next_color, self.get_color_error_delta(next_color)


LUX_LEVELS = 8

RECUR = 5
COL_SPLITS = 8

import numpy as np

LUX_LIMIT = 1


def sum_solid_error(target: TargetImage, color: tuple[int, int, int]) -> np.float64:
    """Get one float representing the error of a solid color."""
    solid = target.get_solid_grid(color)
    return np.sum(target.get_error(solid))


def _replace_background(target: TargetImage, num: int, idx: int = 0) -> None:
    """Replace the default background color with one of the cluster color exemplars.

    :param target: target image to replace the background of
    :param num: number of colors to use
    :param idx: index of the color to use. Will use the largest cluster exemplar by
        default.
    """
    logging.info("replacing default background color")
    cols = target.get_colors(num)
    target.set_background(cols[idx])


def _get_svglayers_instance_from_color(
    target: TargetImage, col: tuple[int, int, int]
) -> SvgLayers:
    """Create an svg layers instance from a TargetImage and a color.

    :param target: target image to use
    :param col: color to use
    :return: svg layers instance

    Write an image representing the gradient (white is better) of how much the given
    color would improve the current state. Use this image to create an SvgLayers
    instance.
    """
    bmp_name = TEMP / f"temp_{'-'.join(map(str, col))}.bmp"
    col_improves = target.get_color_error_delta(col)
    _write_bitmap_from_array(col_improves, bmp_name)
    return SvgLayers(bmp_name, despeckle=1 / 50)


def _get_candidate(
    layers: SvgLayers, col: tuple[int, int, int], lux: float
) -> EtreeElement:
    """Create a candidate element.

    :param layers: svg layers instance to use
    :param col: color to use
    :param lux: lux level to use
    :return: candidate element
    """
    return update_element(
        layers(lux),
        fill=svg_color_tuple(col),
        opacity="0.65",
    )


class ScoredCandidate:

    def __init__(self, color: tuple[int, int, int], layers: SvgLayers, lux: float):
        self.color = color
        self.layers = layers
        self.lux = lux
        self._candidate: EtreeElement | None = None
        self._error: float | None = None

    @property
    def candidate(self) -> EtreeElement:
        if self._candidate is None:
            self._candidate = _get_candidate(self.layers, self.color, self.lux)
        return self._candidate

    @property
    def error(self) -> float:
        if self._error is None:
            self._error = target.get_candidate_error(self.candidate)
        return self._error

    def __lt__(self, other: ScoredCandidate) -> bool:
        return (self.error, self.lux) < (other.error, other.lux)


def _iter_scored_candidates_fixed_color(
    target: TargetImage, col: tuple[int, int, int], num_luxs: int
) -> Iterator[ScoredCandidate]:
    layers = _get_svglayers_instance_from_color(target, col)
    for lux in np.linspace(0, 1, num_luxs):
        scored_candidate = ScoredCandidate(col, layers, lux)
        logging.info(f"  ({lux:0.2f}, {scored_candidate.error})")
        yield scored_candidate


def _iter_scored_candidates(
    target: TargetImage, num_cols: int, num_luxs: int
) -> Iterator[ScoredCandidate]:
    last_color = target.get_last_used_color()
    cols = target.get_colors(num_cols)
    cols = [x for x in cols if get_delta_e(x, last_color) > 10]
    for col in cols:
        logging.info(col)
        yield from _iter_scored_candidates_fixed_color(target, col, num_luxs)


if __name__ == "__main__":
    target = TargetImage(PROJECT / "tests/resources/bird.jpg")
    _replace_background(target, COL_SPLITS)

    idx = 0
    for _ in range(40):
        scored_candidates = tuple(
            _iter_scored_candidates(target, COL_SPLITS, LUX_LEVELS)
        )
        if not scored_candidates:
            break
        best = min(scored_candidates)
        logging.info(f"best: {best.error} {best.color} {best.lux:0.2f}")

        best_color = [x for x in scored_candidates if x.color == best.color]
        logging.info(f"best_color: {len(best_color)}")
        if len(best_color) == 1:
            if best.error > target.sum_error:
                break
            target.append(best.candidate)
            continue

        best, good = sorted(best_color)[:2]
        for _ in range(RECUR):
            logging.info(f"recur: best.score")
            test = ScoredCandidate(best.color, best.layers, (best.lux + good.lux) / 2)
            if test > best:
                break
            best, good = sorted([best, test, good])[:2]
        if best.error > target.sum_error:
            break
        target.append(best.candidate)
            

