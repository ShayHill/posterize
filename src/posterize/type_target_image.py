"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""

import itertools as it
import logging
from pathlib import Path

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
from posterize.paths import PROJECT, WORKING
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

    def get_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the most common colors in the image, largest to smallest
        """
        self.clusters.split_to_at_most(num)
        colors: list[tuple[int, int, int]] = []
        for r, g, b in self.clusters.get_rsorted_exemplars():
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

LUX_LEVELS = 8
RECUR = 5
COL_SPLITS = 8

import numpy as np

LUX_LIMIT = 1


def sum_solid_error(target: TargetImage, color: tuple[int, int, int]) -> np.float64:
    """Get one float representing the error of a solid color."""
    solid = target.get_solid_grid(color)
    return np.sum(target.get_error(solid))


if __name__ == "__main__":
    target = TargetImage(PROJECT / "tests/resources/eyes.jpg")

    logging.info("replacing default background color")
    bg = min(target.get_colors(COL_SPLITS), key=lambda x: sum_solid_error(target, x))
    target.set_background(bg)

    idx = 0
    for _ in range(40):
        target.clusters.split_to_at_most(COL_SPLITS)
        # next_color, next_color_per_pixel = target.evaluate_next_color(idx)
        # next_color_per_pixel = 255 - next_color_per_pixel.astype(np.uint8)
        # _write_bitmap_from_array(next_color_per_pixel, "temp.bmp")

        luxs = np.linspace(0, LUX_LIMIT, LUX_LEVELS)
        cols = [(r, g, b) for r, g, b in target.clusters.get_rsorted_exemplars()]

        # layers = SvgLayers("temp.bmp", despeckle=1 / 50)

        def _scored(
            layers: SvgLayers, lux: float, col: tuple[float, float, float]
        ) -> tuple[float, float, EtreeElement]:
            candidate = layers(lux)
            _ = update_element(candidate, fill=svg_color_tuple(col), opacity="0.65")
            error = target.get_candidate_error(candidate)
            print(error - target.get_candidate_error())
            return error, lux, candidate

        print("scoring lux matrix")

        all_scored: set[tuple[float, float, EtreeElement]] = set()

        for i, col in enumerate(cols):
            print()
            print(f"    scoring color {i+1} of {len(cols)}: {col}", end=" ")
            scored: set[tuple[float, float, EtreeElement]] = set()
            bmp_name = f"temp_{'-'.join(map(str, col))}.bmp"

            col_improves = target.get_color_error_delta(col)
            _write_bitmap_from_array(col_improves, bmp_name)

            layers = SvgLayers(bmp_name, despeckle=1 / 50)
            for lux in luxs:
                print("|", end="")
                scored.add(_scored(layers, lux, col))
            for _ in range(RECUR):
                print("|", end="")
                parent_a, parent_b = sorted(scored, key=lambda x: x[0])[:2]
                new_lux = (parent_a[1] + parent_b[1]) / 2
                scored.add(_scored(layers, new_lux, col))
            layers.close()
            all_scored |= scored

        best = min(all_scored, key=lambda x: x[0])

        if best[0] < target.get_candidate_error():
            print()
            print("    >>> ", f"{best[2].attrib['fill']}", best[1])
            target.append(best[2])
        else:
            break
