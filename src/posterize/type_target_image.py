"""A type to store input images and current error.

This module contains a class to hold an input image and the current state of the
approximation of that image:

* use `set_background` to add a background color
* Use the error attributes to measure candidate svg elements
* append the best candidate using the append method

As elements are appended, the approximation should improve.

:author: Shay Hill
:created: 2024-05-09
"""

from __future__ import annotations

from operator import attrgetter
import os
import copy
from posterize.svg_layers import SvgLayers
from svg_ultralight import update_element
import logging
from svg_ultralight.strings import svg_color_tuple
from pathlib import Path
from typing import Annotated, TypeAlias, Callable, Any
import functools
from posterize import paths

from posterize.image_arrays import write_bitmap_from_array
import numpy as np
import numpy.typing as npt
from basic_colormath import (
    float_tuple_to_8bit_int_tuple,
    get_delta_e,
    rgbs_to_lab,
    get_deltas_e_lab,
    rgb_to_hsv,
)
from cluster_colors import KMedSupercluster, get_image_clusters
from cluster_colors.clusters import Member
from cluster_colors.cut_colors import cut_colors
from cluster_colors.pool_colors import pool_colors
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from PIL import Image
from PIL.Image import Image as ImageType
from svg_ultralight import new_element

from posterize.arrays import normalize_errors_to_8bit
from posterize.rasterize import elem_to_png_array, elem_to_png_bytes, elem_to_png_image

_PixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,3)"]
_LabArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,3)"]
_MonoPixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,1)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,1)"]
_RGBTuple = tuple[int, int, int]


logging.basicConfig(level=logging.INFO)

_SCREEN_HEAD = (
    r'<svg xmlns="http://www.w3.org/2000/svg" '
    r'xmlns:xlink="http://www.w3.org/1999/xlink" '
    r'viewBox="0 0 {0} {1}" width="{0}" height="{1}">'
)
_SCREEN_TAIL = b"</svg>"

_BIG_INT = 2**31 - 1


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


def _load_or_new_root(image: ImageType, path_to_cache: Path | None) -> EtreeElement:
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


def _interp_pixel_arrays(
    pixels_a: _PixelArray, pixels_b: _PixelArray, time: float
) -> _PixelArray:
    """Interpolate between two pixel arrays.

    :param pixels_a: first pixel array
    :param pixels_b: second pixel array
    :param time: weight of the second array [0..1]
    :return: blended pixel array  a * (1 - time) + b * time

    Blend two pixel arrays and re-cast to uint8 with a better distribution across
    [0, 255] than the floor numpy seems to use.
    """
    if time == 0:
        return pixels_a
    if time == 1:
        return pixels_b
    as_floats = pixels_a * (1 - time) + pixels_b * time
    as_floats = np.clip(as_floats / 255, 0, 1)
    as_ints = (as_floats * _BIG_INT).astype(int)
    return (as_ints >> 23).astype(np.uint8)


class TargetImage:
    """A type to store input images, current state, and current cost."""

    def __init__(self, path: Path, path_to_cache: Path | None = None) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param path_to_cache: path to the cache file or None. If given, will cache
            the root element only. Will refresh the cache with each append.
        """
        image = Image.open(path)

        self.shape = (image.width, image.height)
        self._root = _load_or_new_root(image, path_to_cache)
        self._path_to_cache = path_to_cache
        self._clusters = get_image_clusters(path)
        self._image_rgbs = np.array(image).astype(float)
        self._image_labs = rgbs_to_lab(self._image_rgbs)

        # cached state and cost properties
        self.__state_bytes: bytes | None = None
        self.__state_image: ImageType | None = None
        self.__state_rgbs: _PixelArray | None = None
        self.__state_labs: _LabArray | None = None
        self.__cost_array: _ErrorArray | None = None

    @property
    def root(self) -> EtreeElement:
        """Read only root element."""
        return self._root

    # ===========================================================================
    #   Cache state and cost properties until state changes with append.
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
    def state_rgbs(self) -> _PixelArray:
        """Get and cache the current state as an array of pixels."""
        if self.__state_rgbs is None:
            self.__state_rgbs = elem_to_png_array(self.root)
        return self.__state_rgbs

    @property
    def state_labs(self) -> npt.NDArray[np.float64]:
        """Get the current state in LAB color space."""
        if self.__state_labs is None:
            self.__state_labs = rgbs_to_lab(self.state_rgbs)
        return self.__state_labs

    @property
    def state_cost_array(self) -> _ErrorArray:
        """Get the current cost-per-pixel between state and input.

        This should decrease after every append.
        """
        if self.__cost_array is None:
            self.__cost_array = self.get_cost_array(self.state_labs)
        return self.__cost_array

    @property
    def sum_state_cost(self) -> float:
        """Get the sum of the cost at the current state.

        Use this to guarantee any new element will improve the approximation.
        """
        return float(np.sum(self.state_cost_array))

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append

        Update the state and cost grids, update the color clusters with original
        pixel colors weighted by cost, and cache the instance.
        """
        self._root.append(element)

        # reset all cached properties
        self.__state_bytes = None
        self.__state_image = None
        self.__state_rgbs = None
        self.__state_labs = None
        self.__cost_array = None

        # re-weight image colors by cost per pixel
        logging.info("reclustering colors")
        ws = self.state_cost_array[..., np.newaxis]
        needs = np.concatenate([self._image_rgbs, ws], axis=2).reshape(-1, 4)
        self._clusters = _get_clusters(needs)
        logging.info("done reclustering colors")

        # cache self.root
        if self._path_to_cache is None:
            return
        etree.ElementTree(self.root).write(self._path_to_cache)

    # ===========================================================================
    #   Comparisons to and transformations of self._image_array
    # ===========================================================================

    def get_cost_array(self, labs: _LabArray) -> _ErrorArray:
        """Get the cost-per-pixel between a pixel grid and self._image_grid.

        :param pixels: array[n,m,3] grid to compare to self._image_grid
        :return: cost-per-pixel [n,m,1] between grid and self._image_grid

        Values are the Euclidean distance between the pixel grid and the image grid,
        so will range from 0 to (255 * sqrt(3)) = 441.673.

        NOT the *squared* Euclidean distance, because we're going to map cost onto
        grayscale images, and a linear cost gives a better result. The entire
        process is slow due to rasterization, but we aren't sqrt-ing that many cost
        grids, and at typical image sizes, the operation is nearly instantaneous.
        """
        return get_deltas_e_lab(labs, self._image_labs)

    def get_color_cost_delta(self, color: _RGBTuple) -> npt.NDArray[np.float64]:
        """Get the difference in cost between the current state and a solid color.

        :param color: color to compare to the current state
        :return: (w,h,1) array where each value [0..255] is the relative improvement
            of a solid color over the current state. Where result is 0, the current
            state is (probably) better. Where result is 255, the solid color is
            (probably) better. There is a small potential that the current state is
            entirely better or entirely worse, but these cases will be discarded
            later on.
        """
        solid_color = np.full_like(self._image_rgbs, color)
        color_cost = self.get_cost_array(solid_color)
        return color_cost - self.state_cost_array

    def get_color_cost_delta_bifurcated(self, color: _RGBTuple) -> _MonoPixelArray:
        """Get the difference in cost between the current state and a solid color.

        # TODO: review param and return in docstring
        :param color: color to compare to the current state
        :return: ImageType where each pixel is the relative improvement of a solid
            color over the current state. Where pixel is 0, the current state is
            (probably) better. Where pixel is 255, the solid color is (probably)
            better. There is a small potential that the current state is entirely
            better or entirely worse, but these cases will be discarded later on.

        Potrace draws the black pixels, so the output here may be intuitively
        inverted.

        * Any pixel where the color is a better approximation than the current state
          (color cost < state cost -> color cost - state cost < 0) will be 0.

        * Any pixel where the current state is a better approximation than the color
          (color cost > state cost -> color cost - state cost > 0) will be 255.
        """
        delta_array = self.get_color_cost_delta(color)
        delta_array[np.where(delta_array < 0)] = 0
        delta_array[np.where(delta_array > 0)] = 255
        return delta_array.astype(np.uint8)

        return Image.fromarray(self.get_color_cost_delta(color))

    def monochrome_like(self, color: tuple[float, float, float]) -> _PixelArray:
        """Get a solid color array the same shape as self._image_grid.

        :param color: color to make the array
        :return: solid color array
        """
        return np.full_like(self._image_rgbs, color).astype(np.uint8)

    def set_background(self, color: _RGBTuple) -> None:
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

    def _raster_state_with_candidates(self, *candidates: EtreeElement) -> _PixelArray:
        """Get a pixel array of the current state (with potential candidates).

        :param candidates: optional candidates to add to the current state
        :return: pixel array of the current state

        If no candidates are given, the current state (all elements in self.root) is
        returned. If candidates *are* given, these will be layered on top of the
        current state, and cost_grid will be calculated for that arrangement.
        """
        if not candidates:
            return self.state_rgbs

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

    def _get_last_used_color(self) -> _RGBTuple:
        """Get the last color used.

        :return: the color of the last element added
        """
        if len(self.root) == 0:
            msg = "No elements added yet."
            raise ValueError(msg)
        rgb_str = self.root[-1].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = map(int, rgb_str.split(","))
        return (r, g, b)

    def _get_nback_used_color(self, n: int) -> _RGBTuple:
        if len(self.root) < n:
            msg = f"Only {len(self.root)} elements added."
            raise ValueError(msg)
        rgb_str = self.root[-n].attrib["fill"][4:-1]  # '255,255,255'
        r, g, b = map(int, rgb_str.split(","))
        return (r, g, b)

    def get_colors(self, num: int) -> list[_RGBTuple]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the num+1 most common colors in the image

        Return one color that represents the entire self.custers plus the exemplar of
        cluster after splitting to at most num clusters.
        """
        colors = [self._clusters.as_cluster.exemplar]
        if num > 1:
            self._clusters.split_to_at_most(num)
            colors.extend(self._clusters.get_rsorted_exemplars())

        return [float_tuple_to_8bit_int_tuple((r, g, b)) for r, g, b in colors]

    def split_n_back_dist(
        self, colors: list[_RGBTuple], num: int, _n: int = 1
    ) -> list[_RGBTuple]:
        if len(colors) <= num:
            return colors
        if len(self.root) < _n:
            return colors

        avoid = self._get_nback_used_color(_n)

        def dist_from_avoid(color: _RGBTuple) -> float:
            """Get the distance from last_color."""
            return get_delta_e(color, avoid)

        n_keep = max(len(colors) // 2, num)
        colors = sorted(colors, key=dist_from_avoid)[-n_keep:]
        return self.split_n_back_dist(colors, num, _n + 1)

    def get_next_colors(self, max_cols: int, min_cols: int) -> list[_RGBTuple]:
        """Get the most common colors in the image farthest from last_color.

        :param num: number of colors to get (half will be discarded)
        :return: the num // most common colors in the image farthest from last_color

        To increase contrast / interest, select the color of each new elements from a
        group of image colors that are farthest from the color used in the last
        element.
        """
        colors = self.get_colors(max_cols)
        return self.split_n_back_dist(colors, min_cols)

    def _get_candidate_cost_array(self, *candidates: EtreeElement) -> _ErrorArray:
        """Get the cost per pixel of the current state with candidates.

        :param candidates: optional candidates to add to the current state
        :return: cost per pixel of the current state with candidates
        """
        if not candidates:
            return self.state_cost_array
        candidate_array = self._raster_state_with_candidates(*candidates)
        return self.get_cost_array(candidate_array)

    def get_sum_candidate_cost(self, *candidates: EtreeElement) -> float:
        """Get the sum of the cost of the current state with candidates.

        :param candidates: optional candidates to add to the current state
        :return: sum of the cost of the current state with candidates
        """
        return float(np.sum(self._get_candidate_cost_array(*candidates)))


class ScoredCandidate:
    """Setup for calling potrace and comparing the result to the current state.

    You will need several new arrays to create a candidate:

    * _color_array: (w,h,3) the color applied over the entire image at the given
      opacity
    * _color_cost_array: (w,h,1) the cost of the color_array
    * _color_cost_delta: (w,h,1) the difference in cost between the current state and
      the color_array (>0 where current state is better)
    * _color_mask: (w,h,1) 0 where the color_array is better, 255 where the
      current state is better. This will create the bitmap for potrace and also mask
      the color array to determine cost with color applied.
    * _candidate_array: the current state with the color_array applied where it
      improves the approximation (where _potrace_bitmap is 0)
    """

    def __init__(self, target: TargetImage, color: _RGBTuple, opacity: float):
        self._target = target
        self.color = color
        self._opacity = opacity

    def __lt__(self, other: ScoredCandidate) -> bool:
        return self.cost < other.cost

    @functools.cached_property
    def _color_rgbs(self) -> _PixelArray:
        state_array = self._target.state_rgbs
        color_array = np.full_like(state_array, self.color)
        return _interp_pixel_arrays(state_array, color_array, self._opacity)

    @functools.cached_property
    def _color_labs(self) -> _LabArray:
        return rgbs_to_lab(self._color_rgbs)

    @functools.cached_property
    def _color_cost_per_pixel(self) -> npt.NDArray[np.float64]:
        return self._target.get_cost_array(self._color_labs)

    @property
    def _color_cost_delta(self) -> npt.NDArray[np.float64]:
        """Get the difference in cost between the current state and a solid color.

        :param color: color to compare to the current state
        :return: (w,h,1) array where each value [0..255] is the relative improvement
            of a solid color over the current state. Where result is 0, the current
            state is (probably) better. Where result is 255, the solid color is
            (probably) better. There is a small potential that the current state is
            entirely better or entirely worse, but these cases will be discarded
            later on.
        """
        return self._color_cost_per_pixel - self._target.state_cost_array

    @functools.cached_property
    def _color_mask(self) -> _MonoPixelArray:
        """Get the difference in cost between the current state and a solid color.

        # TODO: review param and return in docstring
        :param color: color to compare to the current state
        :return: ImageType where each pixel is the relative improvement of a solid
            color over the current state. Where pixel is 0, the current state is
            (probably) better. Where pixel is 255, the solid color is (probably)
            better. There is a small potential that the current state is entirely
            better or entirely worse, but these cases will be discarded later on.

        Potrace draws the black pixels, so the output here may be intuitively
        inverted.

        * Any pixel where the color is a better approximation than the current state
          (color cost < state cost -> color cost - state cost < 0) will be 0.

        * Any pixel where the current state is a better approximation than the color
          (color cost > state cost -> color cost - state cost > 0) will be 255.
        """
        delta_array = self._color_cost_delta
        delta_array[np.where(delta_array < 0)] = 0
        delta_array[np.where(delta_array > 0)] = 255
        return delta_array.astype(np.uint8)

    @property
    def _layers(self) -> SvgLayers:
        bmp_name = paths.TEMP / f"temp_{'-'.join(map(str, self.color))}.bmp"
        write_bitmap_from_array(self._color_mask, bmp_name)
        return SvgLayers(bmp_name, despeckle=0.02)

    @functools.cached_property
    def candidate(self):
        # TODO: stop context management for svg_layers
        bmp_name = paths.TEMP / f"temp_{'-'.join(map(str, self.color))}.bmp"
        write_bitmap_from_array(self._color_mask, bmp_name)
        svg_layers = SvgLayers(bmp_name, despeckle=0.02)
        candidate = update_element(
            svg_layers(0.5),
            fill=svg_color_tuple(self.color),
            opacity=f"{self._opacity:0.2f}",
        )
        os.unlink(bmp_name)
        return candidate

    @functools.cached_property
    def _costs_where_color_is_better(self) -> npt.NDArray[np.float64]:
        where_color_better = np.where(self._color_mask == 0)
        return self._color_cost_per_pixel[where_color_better]

    @functools.cached_property
    def _costs_where_state_is_better(self) -> npt.NDArray[np.float64]:
        where_state_better = np.where(self._color_mask == 255)
        return self._target.state_cost_array[where_state_better]

    # @functools.cached_property
    # def _candidate_array(self) -> _PixelArray:
    #     logging.info(f"generating cost {self.color}")
    #     candidate_array = np.copy(self._target.state_rgbs)
    #     apply_color = np.where(self._color_mask == 0)
    #     candidate_array[apply_color] = self._color_rgbs[apply_color]
    #     return candidate_array

    @functools.cached_property
    def cost(self) -> float:
        logging.info(f"generating cost {self.color}")
        sum_color_cost = np.sum(self._costs_where_color_is_better)
        sum_state_cost = np.sum(self._costs_where_state_is_better)
        return float(sum_color_cost + sum_state_cost)

    @functools.cached_property
    def sort_key(self) -> float:
        return self.cost * self.max_cost

    @functools.cached_property
    def max_cost(self) -> float:
        logging.info(f"generating max cost {self.color}")
        max_color_cost = max(self._costs_where_color_is_better)
        max_state_cost = max(self._costs_where_state_is_better)
        return max(max_color_cost, max_state_cost)


def _get_infix(arg_val: Path | float) -> str:
    """Format a Path instance, int, or float into a valid filename infix.

    :param arg_val: value to format
    :return: formatted value
    """
    if isinstance(arg_val, Path):
        return arg_val.stem.replace(" ", "_")
    return str(arg_val).replace(".", "p")


def load_target_image(image_path: Path, *args: float | Path) -> TargetImage:
    """Load a cached TargetImage instance or create a new one.

    :param image_path: path to the image
    :param args: additional arguments to use in the cache identifier. If all of these
        are the same, use the cache. If one of these change, create a new instance.
    :return: a TargetImage instance
    """
    cache_identifiers = [_get_infix(x) for x in (image_path, *args)]
    path_to_cache = paths.CACHE / f"{'_'.join(cache_identifiers)}.xml"
    return TargetImage(image_path, path_to_cache=path_to_cache)


def _select_background_color(target: TargetImage, num: int, idx: int = 0) -> None:
    """Replace the default background color with one of the cluster color exemplars.

    :param target: target image to replace the background of
    :param num: number of colors to use
    :param idx: index of the color to use. Will use the largest cluster exemplar by
        default.
    """
    logging.info("replacing default background color")
    cols = target.get_colors(num)
    candidates = [ScoredCandidate(target, col, 1) for col in cols]
    # TODO: restore background selection
    # target.set_background((149, 61, 41))
    target.set_background(min(candidates).color)


def get_posterize_elements(
    image_path: os.PathLike[str],
    max_cols: int,
    min_cols: int,
    opacity: float,
    max_layers: int,
    max_cost_priority: int = 0,
    *,
    do_ignore_cache: bool = False,
) -> EtreeElement:
    """Return a list of elements to create a posterized image.

    :param image_path: path to the image
    :param num_cols: number of colors to split the image into before selecting a
        color for the next element (will add full image exemplar to use num_cols + 1
        colors.
    :param num_luxs: number of lux levels to use
    :param opacity: opacity level to use for candidates
    :param max_layers: maximum number of layers to add to the image

    The first element will be a rect element showing the background color. Each
    additional element will be a path element filled with a color that will
    progressively improve the approximation.
    """
    if do_ignore_cache:
        target = TargetImage(Path(image_path))
    else:
        target = load_target_image(
            Path(image_path), max_cols, min_cols, opacity, max_cost_priority
        )

    if len(target.root) < 1:
        _select_background_color(target, max_cols)
        target.render_state(paths.WORKING / "state")

    while len(target.root) < max_layers:
        logging.info(f"layer {len(target.root) + 1}")
        if max_cost_priority and (len(target.root) - 1) % max_cost_priority == 0:
            cols = target.get_next_colors(max_cols, max(min_cols, max_cols // 2))
            sort_key = attrgetter("max_cost")
        else:
            cols = target.get_next_colors(max_cols, min_cols)
            sort_key = attrgetter("sort_key")
        candidates = [ScoredCandidate(target, col, opacity) for col in cols]
        best_candidate = min(candidates, key=sort_key)

        if best_candidate.cost > target.sum_state_cost:
            logging.info("no more improvement found")
            break
        logging.info(f"best candidate: {best_candidate.color}")
        logging.info(f"best cost: {best_candidate.cost}")
        target.append(best_candidate.candidate)
        target.render_state(paths.WORKING / "state")

    return target.root


if __name__ == "__main__":
    _ = get_posterize_elements(
        paths.PROJECT / "tests/resources/adidas.jpg",
        12,
        3,
        0.85,
        48,
        0,
        do_ignore_cache=True,
    )
