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

import copy
import logging
from pathlib import Path
from typing import Annotated, TypeAlias

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
from svg_ultralight import new_element

from posterize.arrays import normalize_errors_to_8bit
from posterize.rasterize import elem_to_png_array, elem_to_png_bytes, elem_to_png_image

_PixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,3)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,1)"]


logging.basicConfig(level=logging.INFO)

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


class TargetImage:
    """A type to store input images, current state, and current error."""

    def __init__(self, path: Path, path_to_cache: Path | None = None) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param path_to_cache: path to the cache file or None. If given, will cache
            the root element only. Will refresh the cache with each append.
        """
        image = Image.open(path)

        self._root = _load_or_new_root(image, path_to_cache)
        self._path_to_cache = path_to_cache
        self._clusters = get_image_clusters(path)
        self._image_array = np.array(image).astype(float)

        # cached state and error properties
        self.__state_bytes: bytes | None = None
        self.__state_image: ImageType | None = None
        self.__state_array: _PixelArray | None = None
        self.__error_array: _ErrorArray | None = None

    @property
    def root(self) -> EtreeElement:
        """Read only root element."""
        return self._root

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
    def _state_array(self) -> _PixelArray:
        """Get and cache the current state as an array of pixels."""
        if self.__state_array is None:
            self.__state_array = elem_to_png_array(self.root)
        return self.__state_array

    @property
    def _error_array(self) -> _ErrorArray:
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
        self._root.append(element)

        # reset all cached properties
        self.__state_bytes = None
        self.__state_image = None
        self.__state_array = None
        self.__error_array = None

        # re-weight image colors by error per pixel
        ws = self._error_array[..., np.newaxis]
        needs = np.concatenate([self._image_array, ws], axis=2).reshape(-1, 4)
        self._clusters = _get_clusters(needs)

        # cache self.root
        if self._path_to_cache is None:
            return
        etree.ElementTree(self.root).write(self._path_to_cache)

    # ===========================================================================
    #   Comparisons to and transformations of self._image_array
    # ===========================================================================

    def get_error(self, pixels: _PixelArray) -> _ErrorArray:
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
        """Get the difference in error between the current state and a solid color.

        :param color: color to compare to the current state
        :return: (w,h,1) array where each value [0..255] is the relative improvement
            of a solid color over the current state. Where result is 0, the current
            state is (probably) better. Where result is 255, the solid color is
            (probably) better. There is a small potential that the current state is
            entirely better or entirely worse, but these cases will be discarded
            later on.
        """
        solid_color = np.full_like(self._image_array, color)
        color_error = self.get_error(solid_color)
        error_delta = color_error - self._error_array
        return normalize_errors_to_8bit(error_delta)

    def monochrome_like(self, color: tuple[float, float, float]) -> _PixelArray:
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

    def _raster_state_with_candidates(self, *candidates: EtreeElement) -> _PixelArray:
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
        colors = [self._clusters.as_cluster.exemplar]
        if num > 1:
            self._clusters.split_to_at_most(num)
            colors.extend(self._clusters.get_rsorted_exemplars())

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

    def _get_candidate_error_array(self, *candidates: EtreeElement) -> _ErrorArray:
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
