"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""

from __future__ import annotations
import numpy as np
import os
from cairosvg import svg2png
from tempfile import NamedTemporaryFile
import pickle
import itertools as it
from contextlib import suppress
import warnings
import io
import logging
from pathlib import Path
import dataclasses
from svg_ultralight.inkscape import rasterize_svg
import inspect
import copy
import PIL


from typing import Iterator, TypeVar, cast
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
from PIL.Image import Image as ImageType
from svg_ultralight import new_element, update_element, write_root
from svg_ultralight.strings import svg_color_tuple

from posterize.color_dist import desaturate_by_color
from posterize.image_arrays import (
    _write_bitmap_from_array,
    get_image_pixels,
    write_monochrome_bmp,
)
from posterize.paths import PROJECT, TEMP, WORKING, CACHE
from posterize.svg_layers import SvgLayers
from posterize.arrays import normalize_errors_to_8bit

logging.basicConfig(level=logging.INFO)

_T = TypeVar("_T")

_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")

_SCREEN_HEAD = (
    r'<svg xmlns="http://www.w3.org/2000/svg" '
    r'xmlns:xlink="http://www.w3.org/1999/xlink" '
    r'viewBox="0 0 {0} {1}" width="{0}" height="{1}">'
)
_SCREEN_TAIL = b"</svg>"

def _get_clusters(colors: npt.NDArray[np.float64]) -> KMedSupercluster:
    """Pool and cut colors.

    :param colors: colors to pool and cut
    :return: pooled and cut colors

    Trim the number of colors in an image to a manageable number (512), then create a
    KMedSupercluster instance.
    """
    pooled = pool_colors(colors)
    pooled_and_cut = cut_colors(pooled, 512)
    members = Member.new_members(pooled_and_cut)
    return KMedSupercluster(members)

def _get_singleton_value(set_: set[_T]) -> _T:
    """Get the single value from a set.

    :param set_: set to get the single value from
    :return: the single value from the set

    Raise a ValueError if the set does not contain exactly one value.
    """
    value, = set_
    return value


def load_or_new_root(image: ImageType, path_to_cache: Path | None) -> EtreeElement:
    """Load a cached root or create a new one.

    :param image: image to get the dimensions from
    :param path_to_cache: path to the cache file or None
    """
    if path_to_cache is not None and path_to_cache.exists():
        root = etree.parse(path_to_cache).getroot()
        logging.info(f"loaded cache '{path_to_cache.name}' with {len(root)} elements.")
        return root
    bhead = _SCREEN_HEAD.format(image.width, image.height).encode()
    return etree.fromstring(bhead + _SCREEN_TAIL)


class TargetImage:
    """A type to store input images and current error."""

    def __init__(
        self,
        path: Path,
        path_to_cache: Path | None = None,
    ) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param lux: number of quantiles to use in error calculation
        """
        image = Image.open(path)
        self.root = load_or_new_root(image, path_to_cache)
        self.path_to_cache = path_to_cache
        self.clusters = get_image_clusters(path)

        self._image_grid = np.array(image).astype(float)
        self.__state_grid: npt.NDArray[np.uint8] | None = None
        self.__error_grid: npt.NDArray[np.float64] | None = None
        self.__state_png: ImageType | None = None

    @property
    def _state_grid(self) -> npt.NDArray[np.uint8]:
        """Get the current state grid."""
        if self.__state_grid is None:
            self.__state_grid = self._raster_state()
        return self.__state_grid

    @property
    def _error_grid(self) -> npt.NDArray[np.float64]:
        """Get the current error grid."""
        if self.__error_grid is None:
            self.__error_grid = self.get_error(self._state_grid)
        return self.__error_grid

    # def __render_svg_and_return_path(self, filename: Path | None = None, num: int | None = None) -> Path:
    #     if num:
    #         root = self.root
    #     else:
    #         root = copy.deepcopy(self.root)
    #         subelements = root[:num]
    #         root.clear()
    #         for subelement in subelements:
    #             root.append(subelement)
    #     if filename is None:
    #         with NamedTemporaryFile(mode="wb", suffix=".svg", delete=False) as f:
    #             filename = Path(f.name)
    #     _ = write_root(_INKSCAPE, filename, root)
    #     return filename

    # def _render_svg(self, filename: Path | None, num: int | None = None) -> Path | None:
    #     num = num or len(self.root)
    #     if filename:
    #         filename = filename.parent / f"{filename.stem}_{num:03n}.svg"
    #         return self.__render_svg_and_return_path(filename, num)
    #     filename = self.__render_svg_and_return_path(None, num)
    #     os.unlink(filename)

    @property
    def _state_png(self, num: int | None = None) -> ImageType:
        """Get the current state as a PNG."""
        if self.__state_png is None:
            png_bytes = cast(None | bytes, svg2png(etree.tostring(self.root)))
            if not isinstance(png_bytes, bytes):
                msg = "failed to render PNG."
                raise ValueError(msg)
            self.__state_png = Image.open(io.BytesIO(png_bytes))
        return self.__state_png

    @property
    def sum_error(self) -> float:
        """Get the sum of the error at the current state."""
        return float(np.sum(self._error_grid))

    def get_colors(self, num: int = 0) -> set[tuple[int, int, int]]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the num+1 most common colors in the image, largest to smallest

        Return one color that represents the entire self.custers plus the exemplar of
        cluster after splitting to at most num colors.
        """
        colors = {self.clusters.as_cluster.exemplar}
        if num > 1:
            self.clusters.split_to_at_most(num)
            colors |= set(self.clusters.get_rsorted_exemplars())
        return {float_tuple_to_8bit_int_tuple((r, g, b)) for r, g, b in colors}

    def get_next_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image farthest from last_color.

        :param num: number of colors to get (half will be discarded)
        :return: the num // most common colors in the image farthest from last_color
        """
        last_color = self.get_last_used_color()
        colors = self.get_colors(num)
        return sorted(colors, key=lambda x: get_delta_e(x, last_color))[-num // 2 :]

    def set_background(self, color: tuple[int, int, int]) -> None:
        """Replace the background with the current background color."""
        if len(self.root) != 0:
            msg = "Background color must be set before adding elements."
            raise ValueError(msg)
        bg = new_element("rect", width="100%", height="100%", fill=f"rgb{color}")
        self.append(bg)

    def get_solid_grid(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """Get a solid color grid the same shape as self._image_grid.

        :param color: color to make the grid
        :return: solid color grid
        """
        return np.full_like(self._image_grid, color).astype(np.uint8)

    def get_error(self, grid: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        """Get the error-per-pixel between a pixel grid and self._image_grid.

        :param grid: array[n,m,3] grid to compare to self._image_grid
        :return: error-per-pixel [n,m,1] between grid and self._image_grid
        """
        image_as_floats = self._image_grid.astype(float)
        grid_as_floats = grid.astype(float)
        return np.sum((image_as_floats - grid_as_floats) ** 2, axis=2) ** (1 / 2)

    def _get_candidate_error_grid(
        self, candidate: EtreeElement | None = None
    ) -> npt.NDArray[np.float64]:
        if candidate is None:
            state_grid = self._state_grid
        else:
            state_grid = self._raster_state(candidate)
        return self.get_error(state_grid)

    def get_candidate_error(self, candidate: EtreeElement | None = None) -> float:
        return float(np.sum(self._get_candidate_error_grid(candidate)))

    def get_last_used_color(self) -> tuple[int, int, int]:
        """Get the last color used.

        :return: the color of the last element added
        """
        if len(self.root) == 0:
            msg = "No elements added yet."
            raise ValueError(msg)
        rgb_str = self.root[-1].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = map(int, rgb_str.split(","))
        return (r, g, b)

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append

        Update the state and error grids, update the color clusters with original
        pixel colors weighted by error, and cache the instance.
        """
        self.root.append(element)
        self.__state_grid = None
        self.__error_grid = None
        self.__state_png = None

        ws = self._error_grid[..., np.newaxis]
        needs = np.concatenate([self._image_grid, ws], axis=2).reshape(-1, 4)
        self.clusters = _get_clusters(needs)

        if self.path_to_cache is None:
            return
        with (self.path_to_cache).open("wb") as f:
            _ = f.write(etree.tostring(self.root))


    def _raster_state(self, *candidates: EtreeElement) -> npt.NDArray[np.uint8]:
        root = copy.deepcopy(self.root)
        for candidate in candidates:
            root.append(candidate)
        png_bytes = cast(None | bytes, svg2png(etree.tostring(root)))
        png_image = Image.open(io.BytesIO(png_bytes))
        return np.array(png_image).astype(np.uint8)[:, :, :3]




    def get_color_error_delta(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """What is the difference in error between the current state and a solid color?"""
        solid_color = np.full_like(self._image_grid, color)
        color_error = self.get_error(solid_color)
        error_delta = color_error - self._error_grid
        return normalize_errors_to_8bit(error_delta)


def _get_sum_solid_error(
    target: TargetImage, color: tuple[int, int, int]
) -> np.float64:
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
    scored = [(_get_sum_solid_error(target, col), col) for col in cols]
    target.set_background(min(scored)[1])


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
    layers: SvgLayers, col: tuple[int, int, int], lux: float, opacity: float
) -> EtreeElement:
    """Create a candidate element.

    :param layers: svg layers instance to use
    :param col: color to use
    :param lux: lux level to use
    :param opacity: opacity level to use
    :return: candidate element
    """
    return update_element(
        layers(lux),
        fill=svg_color_tuple(col),
        opacity=f"{opacity:0.2f}",
    )


class ScoredCandidate:
    """A candidate svg path with the arguments necessary to produce it."""

    def __init__(
        self,
        target: TargetImage,
        color: tuple[int, int, int],
        layers: SvgLayers,
        lux: float,
        opacity: float,
    ):
        """Initialize a ScoredCandidate.

        :param target: target image for calculating error
        :param color: color to use
        :param layers: SvgLayers instance to use (created externally from color)
        :param lux: lux level to use
        :param opacity: opacity level to use for candidate
        """
        self.target = target
        self.color = color
        self.layers = layers
        self.lux = lux
        self.opacity = opacity
        self._candidate: EtreeElement | None = None
        self._error: float | None = None

    @property
    def candidate(self) -> EtreeElement:
        """Create the candidate element."""
        if self._candidate is None:
            self._candidate = _get_candidate(
                self.layers, self.color, self.lux, self.opacity
            )
        return self._candidate

    @property
    def error(self) -> float:
        """Calculate the error of the candidate."""
        if self._error is None:
            self._error = self.target.get_candidate_error(self.candidate)
        return self._error

    def __lt__(self, other: ScoredCandidate) -> bool:
        """Compare two ScoredCandidates by error and lux.

        The best candidate is the candidate with the lowest error. A tie is unlikely,
        but if it happens, choose the larger lux level (which presumably will add the
        most geometry).
        """
        return (self.error, -self.lux) < (other.error, -other.lux)


def _iter_scored_candidates_fixed_color(
    target: TargetImage, col: tuple[int, int, int], num_luxs: int, opacity: float
) -> Iterator[ScoredCandidate]:
    """Iterate over scored candidates for a fixed color.

    :param target: target image against which to score the candidates
    :param col: color to use
    :param num_luxs: number of lux levels to use
    :param opacity: opacity level to use for candidate

    To  create candidates, an SvgLayers instance is created from each color. This is
    an expensive operation, so do it once and then query it for each lux level.
    """
    layers = _get_svglayers_instance_from_color(target, col)
    for lux in np.linspace(0, 1, num_luxs):
        scored_candidate = ScoredCandidate(target, col, layers, lux, opacity)
        # logging.info(f"  ({lux:0.2f}, {scored_candidate.error})")
        yield scored_candidate


def _iter_scored_candidates(
    target: TargetImage, num_cols: int, num_luxs: int, opacity: float
) -> Iterator[ScoredCandidate]:
    """Iterate over scored candidates.

    :param target: target image against which to score the candidates
    :param num_cols: number of colors to use
    :param num_luxs: number of lux levels to use
    :param opacity: opacity level to use for candidate

    Yield a ScoredCandidate instance for each intersection of colors and lux levels.
    """
    cols = target.get_next_colors(num_cols)
    for col in cols:
        logging.info(col)
        yield from _iter_scored_candidates_fixed_color(target, col, num_luxs, opacity)


def _get_infix(arg_val: Path | float) -> str:
    """Format a Path instance, int, or float into a valid filename infix.

    :param arg_val: value to format
    :return: formatted value
    """
    if isinstance(arg_val, Path):
        return arg_val.stem
    return str(arg_val).replace(".", "p")


def load_target_image(image_path: Path) -> TargetImage:
    """Load a cached TargetImage instance or create a new one.

    :param image_path: path to the image
    :return: a TargetImage instance
    """
    frame = inspect.currentframe()
    if frame is None:
        msg = "Failed to load frame from current function."
        raise RuntimeError(msg)
    frame = frame.f_back
    if frame is None:
        msg = "Failed to load frame from calling function."
        raise RuntimeError(msg)
    args_info = inspect.getargvalues(frame)
    caller_args = [_get_infix(args_info.locals[arg]) for arg in args_info.args][:-1]
    path_to_cache = CACHE / f"{'_'.join(caller_args)}.xml"
    return TargetImage(image_path, path_to_cache=path_to_cache)


def get_posterize_elements(
    image_path: os.PathLike[str],
    num_cols: int,
    num_luxs: int,
    opacity: float,
    max_layers: int,
) -> list[EtreeElement]:
    """Return a list of elements to create a posterized image.

    :param image_path: path to the image
    :param num_cols: number of colors to split the image into before selecting a
        color for the next element (will add full image exemplar to use num_cols + 1
        colors.
    :param num_luxs: number of lux levels to use
    :param opacity: opacity level to use for candidate
    :param max_layers: maximum number of layers to add to the image

    The first element will be a rect element showing the background color. Each
    additional element will be a path element filled with a color that will
    progressively improve the approximation.
    """
    target = load_target_image(Path(image_path))

    if len(target.root) < 1:
        _replace_background(target, num_cols)

    while len(target.root) < max_layers:
        logging.info(f"layer {len(target.root) + 1}")
        scored_candidates = list(
            _iter_scored_candidates(target, num_cols, num_luxs, opacity)
        )
        if not scored_candidates:
            # shouldn't happen, but cover pathological argument values
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
        for _ in range(5):
            logging.info(f"recur: {best.lux} {best.error}")
            test = ScoredCandidate(
                target, best.color, best.layers, (best.lux + good.lux) / 2, opacity
            )
            if test > good:
                logging.info("breaking due to high test score")
                break
            best, good = sorted([best, test, good])[:2]
        if best.error > target.sum_error:
            break
        target.append(best.candidate)
    return target.root


if __name__ == "__main__":
    _ = get_posterize_elements(PROJECT / "tests/resources/taleb.jpg", 4, 8, 0.85, 40)
