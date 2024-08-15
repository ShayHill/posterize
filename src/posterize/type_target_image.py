"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""

from __future__ import annotations
import numpy as np
import os
import pickle
import itertools as it
import io
import logging
from pathlib import Path
import dataclasses
from svg_ultralight.inkscape import rasterize_svg
import inspect
import copy


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
from posterize.paths import PROJECT, TEMP, WORKING, CACHE
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
    outliers). If an image has a lot of background (> 50% pixels are the same color),
    it will be necessary to stretch the interquartile range to have a range at all.
    """
    bot_percent = 25
    top_percent = 75
    q25 = q75 = 0
    while bot_percent >= 0:
        q25 = np.quantile(floats, bot_percent / 100)
        q75 = np.quantile(floats, top_percent / 100)
        if q25 < q75:
            break
        bot_percent -= 5
        top_percent += 5
    else:
        # image is solid color
        return np.float64(q25 - 1), np.float64(q25)

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

    def __init__(
        self,
        path: Path,
        cache_filename: Path | None = None,
    ) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param lux: number of quantiles to use in error calculation
        """
        image = Image.open(path)

        if cache_filename is not None and cache_filename.exists():
            with cache_filename.open("rb") as f:
                self.root = etree.fromstring(f.read())
                print(f"loaded {cache_filename} with {len(self.root)} elements")
        else:
            bhead = SCREEN_HEAD.format(image.width, image.height).encode()
            self.root = etree.fromstring(bhead + b"\n" + SCREEN_TAIL)


        self.path = path
        self.cache_filename = cache_filename

        image = Image.open(path)
        self._bhead = SCREEN_HEAD.format(image.width, image.height).encode()

        self.clusters = get_image_clusters(path)
        bg_color = self.get_colors(1)[0]

        self._image_grid = np.array(image).astype(float)
        self.set_background(bg_color)
        self._state_grid = self.get_solid_grid(bg_color)
        self._error_grid = self.get_error(self._state_grid)

    @property
    def sum_error(self) -> float:
        return float(np.sum(self._error_grid))

    def get_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the num+1 most common colors in the image, largest to smallest
        """
        self.clusters.split_to_at_most(num)
        num_cols = self.clusters.get_rsorted_exemplars()
        main_col = self.clusters.as_cluster.exemplar
        colors = [main_col, *num_cols]
        return [float_tuple_to_8bit_int_tuple((r, g, b)) for r, g, b in colors]

    def get_next_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image farthest from last_color."""
        last_color = self.get_last_used_color()
        colors = set(self.get_colors(num)) | set(self.get_colors(1))
        return sorted(colors, key=lambda x: get_delta_e(x, last_color))[-num // 2 :]


    def set_background(self, color: tuple[int, int, int]) -> None:
        """Replace the background with the current background color."""
        bg_element = new_element(
            "rect", width="100%", height="100%", fill=f"rgb{color}"
        )
        try:
            self.root[0] = bg_element
        except IndexError:
            self.root.append(bg_element)
        self._state_grid = self._raster_state(None, WORKING / "state.svg")

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
        image_as_floats = self._image_grid.astype(float)
        grid_as_floats = grid.astype(float)
        return np.sum((image_as_floats - grid_as_floats) ** 2, axis=2) ** (1 / 2)

    def _get_candidate_error_grid(
        self, candidate: EtreeElement | None = None
    ) -> npt.NDArray[np.float64]:
        if candidate is None:
            state_grid = self._state_grid
        else:
            state_grid = self._raster_state(candidate, WORKING / "candidate.svg")
        return self.get_error(state_grid)

    def get_candidate_error(self, candidate: EtreeElement | None = None) -> float:
        return float(np.sum(self._get_candidate_error_grid(candidate)))

    def get_last_used_color(self) -> tuple[int, int, int]:
        """Get the last color used.

        :return: the color of the last element added
        """
        rgb_str = self.root[-1].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = (int(x) for x in rgb_str.split(","))
        return (r, g, b)

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append

        Update the state and error grids, update the color clusters with original
        pixel colors weighted by error, and cache the instance.
        """
        self.root.append(element)
        self._state_grid = self._raster_state(None, WORKING / "state.svg")
        self._error_grid = self._get_candidate_error_grid()

        ws = self._error_grid[..., np.newaxis]
        needs = np.concatenate([self._image_grid, ws], axis=2).reshape(-1, 4)
        self.clusters = get_clusters(needs)

        if self.cache_filename is None:
            return
        with (self.cache_filename).open("wb") as f:
            _ = f.write(etree.tostring(self.root))

    def _raster_state(
        self, candidate: EtreeElement | None, filename: Path
    ) -> npt.NDArray[np.uint8]:
        root = copy.deepcopy(self.root)
        if candidate is not None:
            root.append(candidate)
        _ = write_root(_INKSCAPE, filename, root)
        png_bytes = rasterize_svg(_INKSCAPE, filename)
        png_image = Image.open(io.BytesIO(png_bytes))
        # TODO: make this into something besides a guess based on filename
        if "state" in filename.stem:
            png_image.save(WORKING / f"state{len(self.root):02n}.png")
        return np.array(png_image).astype(np.uint8)[:, :, :3]

    def get_color_error_delta(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """What is the difference in error between the current state and a solid color?"""
        solid_color = np.full_like(self._image_grid, color)
        color_error = self.get_error(solid_color)
        error_delta = color_error - self._error_grid
        return _normalize_errors_to_8bit(error_delta)



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
    cache_filename = CACHE / f"{'_'.join(caller_args)}.xml"
    return TargetImage(image_path, cache_filename=cache_filename)


def dump_target_image(target: TargetImage) -> None:
    """Dump a TargetImage instance to a cache file.

    :param target: target image to dump
    """
    if target.cache_filename is None:
        return
    with (CACHE / target.cache_filename).open("wb") as file:
        pickle.dump(target, file)


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

    if len(target.root) < 2:
        logging.info(f"old_bg_color: {target.get_last_used_color()}")
        _replace_background(target, num_cols)
        logging.info(f"new_bg_color: {target.get_last_used_color()}")

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
    _ = get_posterize_elements(
        PROJECT / "tests/resources/taleb.jpg", 3, 8, 0.85, 40
    )
