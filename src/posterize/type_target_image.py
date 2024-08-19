"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from basic_colormath import float_tuple_to_8bit_int_tuple, get_delta_e
from cluster_colors import KMedSupercluster, get_image_clusters
from cluster_colors.clusters import Member
from cluster_colors.cut_colors import cut_colors
from cluster_colors.pool_colors import pool_colors
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from PIL import Image
from PIL.Image import Image as ImageType
from svg_ultralight import new_element, update_element
from svg_ultralight.strings import svg_color_tuple

from posterize.arrays import normalize_errors_to_8bit
from posterize.image_arrays import _write_bitmap_from_array
from posterize.paths import CACHE, PROJECT, TEMP, WORKING
from posterize.rasterize import elem_to_png_array, elem_to_png_bytes, elem_to_png_image
from posterize.svg_layers import SvgLayers

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


def _slice_elem(elem: EtreeElement, num: int | None = None) -> EtreeElement:
    """Get a copy of an element with a limited number of children.

    :param elem: element to copy
    :param num: number of children to copy
    :return: a copy of the element with a limited number of children

    If the element has more children than the limit, return a copy of the element
    with the first limit children.
    """
    copy_of_elem = copy.deepcopy(elem)
    if num is None:
        return copy_of_elem
    for subelement in copy_of_elem[num:]:
        copy_of_elem.remove(subelement)
    return copy_of_elem


def _get_singleton_value(set_: set[_T]) -> _T:
    """Get the single value from a set.

    :param set_: set to get the single value from
    :return: the single value from the set

    Raise a ValueError if the set does not contain exactly one value.
    """
    (value,) = set_
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

    def __init__(self, path: Path, path_to_cache: Path | None = None) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param lux: number of quantiles to use in error calculation
        """
        image = Image.open(path)
        self.root = load_or_new_root(image, path_to_cache)
        self.path_to_cache = path_to_cache
        self.clusters = get_image_clusters(path)

        self._image_array = np.array(image).astype(float)

        # cached state and error properties
        self.__state_bytes: bytes | None = None
        self.__state_image: ImageType | None = None
        self.__state_array: npt.NDArray[np.uint8] | None = None
        self.__error_array: npt.NDArray[np.float64] | None = None

    # ===========================================================================
    #   Cache state and error properties until state changes with append.
    # ===========================================================================

    @property
    def _state_bytes(self) -> bytes:
        """Get and cache the current state as PNG bytes."""
        if self.__state_bytes is None:
            self.__state_bytes = elem_to_png_bytes(self.root)
        return self.__state_bytes

    @property
    def _state_image(self) -> ImageType:
        """Get and cache the current state as a PIL.Image.Image png instance."""
        if self.__state_image is None:
            self.__state_image = elem_to_png_image(self.root)
        return self.__state_image

    @property
    def _state_array(self) -> npt.NDArray[np.uint8]:
        """Get and cache the current state as an array of pixels."""
        if self.__state_array is None:
            self.__state_array = elem_to_png_array(self.root)
        return self.__state_array

    @property
    def _error_array(self) -> npt.NDArray[np.float64]:
        """Get the current error-per-pixel between state and input."""
        if self.__error_array is None:
            self.__error_array = self.get_error(self._state_array)
        return self.__error_array

    @property
    def sum_error(self) -> float:
        """Get the sum of the error at the current state.

        Use this to guarantee any new element will improve the approximation.
        """
        return float(np.sum(self._error_array))

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append

        Update the state and error grids, update the color clusters with original
        pixel colors weighted by error, and cache the instance.
        """
        self.root.append(element)

        # reset all cached properties
        self.__state_bytes = None
        self.__state_image = None
        self.__state_array = None
        self.__error_array = None

        # re-weight image colors by error per pixel
        ws = self._error_array[..., np.newaxis]
        needs = np.concatenate([self._image_array, ws], axis=2).reshape(-1, 4)
        self.clusters = _get_clusters(needs)

        # cache self.root
        if self.path_to_cache is None:
            return
        etree.ElementTree(self.root).write(self.path_to_cache)

    # ===========================================================================
    #   Comparisons to and transformations of self._image_array
    # ===========================================================================

    def get_error(self, pixels: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        """Get the error-per-pixel between a pixel grid and self._image_grid.

        :param pixels: array[n,m,3] grid to compare to self._image_grid
        :return: error-per-pixel [n,m,1] between grid and self._image_grid

        Values are the Euclidean distance between the pixel grid and the image grid,
        so will range from 0 to (255 * sqrt(3)) = 441.673.

        NOT the *squared* Euclidean distance, because we're going to map error onto
        grayscale images, and a linear error gives a better result. The entire
        process is slow due to rasterization, but we aren't sqrt-ing that many error
        grids, and at typical image sizes, the operation is nearly instantaneous.
        """
        image_as_floats = self._image_array.astype(float)
        grid_as_floats = pixels.astype(float)
        return np.sum((image_as_floats - grid_as_floats) ** 2, axis=2) ** 0.5

    def get_color_error_delta(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """Get the difference in error between the current state and a solid color."""
        solid_color = np.full_like(self._image_array, color)
        color_error = self.get_error(solid_color)
        error_delta = color_error - self._error_array
        return normalize_errors_to_8bit(error_delta)

    def monochrome_like(
        self, color: tuple[float, float, float]
    ) -> npt.NDArray[np.uint8]:
        """Get a solid color array the same shape as self._image_grid.

        :param color: color to make the array
        :return: solid color array
        """
        return np.full_like(self._image_array, color).astype(np.uint8)

    def set_background(self, color: tuple[int, int, int]) -> None:
        """Add an opaque rectangle.

        :param color: color to use
        """
        if len(self.root) != 0:
            msg = "Background color must be set before adding elements."
            raise ValueError(msg)
        bg = new_element("rect", width="100%", height="100%", fill=f"rgb{color}")
        self.append(bg)

    # ===========================================================================
    #   Render state to disk
    # ===========================================================================

    def _raster_state_with_candidates(
        self, *candidates: EtreeElement
    ) -> npt.NDArray[np.uint8]:
        """Get a pixel array of the current state (with potential candidates).

        :param candidates: optional candidates to add to the current state
        :return: pixel array of the current state

        If no candidates are given, the current state (all elements in self.root) is
        returned. If candidates *are* given, these will be layered on top of the
        current state, and error_grid will be calculated for that arrangement.
        """
        if not candidates:
            return self._state_array

        root = _slice_elem(self.root)
        for candidate in candidates:
            root.append(candidate)
        return elem_to_png_array(root)

    def render_state(
        self, output_path: Path, num: int | None = None, *, do_png: bool = False
    ):
        """Write the current state (or a past state) to disk as svg and png files.

        :param path_stem: path to write the files to
            path.with_suffix(".svg") will be the svg file
            path.with_suffix(".png") will be the png file
        :param num: optional number to select a past state. If None, the current state
            will be written. If a number is given, the state with num elements will
            be written as stem_{num:03n}.svg and stem_{num:03n}.png.
        """
        if num is None:
            root_svg = self.root
            root_png = self._state_image
        else:
            root_svg = _slice_elem(self.root)
            root_png = elem_to_png_image(root_svg)
            output_path = output_path.parent / f"{output_path.stem}_{num:03n}"

        svg_filename = output_path.with_suffix(".svg")
        png_filename = output_path.with_suffix(".png")

        logging.info(f"writing {output_path.stem}")
        etree.ElementTree(root_svg).write(svg_filename)
        if do_png:
            root_png.save(png_filename)

    # ===========================================================================
    #   Query and update the color clusters
    # ===========================================================================

    def _get_last_used_color(self) -> tuple[int, int, int]:
        """Get the last color used.

        :return: the color of the last element added
        """
        if len(self.root) == 0:
            msg = "No elements added yet."
            raise ValueError(msg)
        rgb_str = self.root[-1].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = map(int, rgb_str.split(","))
        return (r, g, b)

    def get_colors(self, num: int = 0) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the num+1 most common colors in the image

        Return one color that represents the entire self.custers plus the exemplar of
        cluster after splitting to at most num clusters.
        """
        colors = [self.clusters.as_cluster.exemplar]
        if num > 1:
            self.clusters.split_to_at_most(num)
            colors.extend(self.clusters.get_rsorted_exemplars())

        return [float_tuple_to_8bit_int_tuple((r, g, b)) for r, g, b in colors]

    def get_next_colors(self, num: int) -> list[tuple[int, int, int]]:
        """Get the most common colors in the image farthest from last_color.

        :param num: number of colors to get (half will be discarded)
        :return: the num // most common colors in the image farthest from last_color

        To increase contrast / interest, select the color of each new elements from a
        group of image colors that are farthest from the color used in the last
        element.
        """
        colors = self.get_colors(num)
        if len(self.root) == 0:
            return colors

        last_color = self._get_last_used_color()

        def dist_from_last(color: tuple[int, int, int]) -> float:
            """Get the distance from last_color."""
            return get_delta_e(color, last_color)

        return sorted(colors, key=dist_from_last)[-num // 2 :: -1]

    def _get_candidate_error_array(
        self, *candidates: EtreeElement
    ) -> npt.NDArray[np.float64]:
        """Get the error per pixel of the current state with candidates.

        :param candidates: optional candidates to add to the current state
        :return: error per pixel of the current state with candidates
        """
        if not candidates:
            return self._error_array
        candidate_array = self._raster_state_with_candidates(*candidates)
        return self.get_error(candidate_array)

    def get_sum_candidate_error(self, *candidates: EtreeElement) -> float:
        """Get the sum of the error of the current state with candidates.

        :param candidates: optional candidates to add to the current state
        :return: sum of the error of the current state with candidates
        """
        return float(np.sum(self._get_candidate_error_array(*candidates)))


def _get_sum_solid_error(
    target: TargetImage, color: tuple[int, int, int]
) -> np.float64:
    """Get one float representing the error of a solid color."""
    solid = target.monochrome_like(color)
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
        layers(lux), fill=svg_color_tuple(col), opacity=f"{opacity:0.2f}"
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
            self._error = self.target.get_sum_candidate_error(self.candidate)
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
        target.render_state(WORKING / "state.svg")
    return target.root


if __name__ == "__main__":
    _ = get_posterize_elements(PROJECT / "tests/resources/bird.jpg", 3, 4, 0.35, 40)
