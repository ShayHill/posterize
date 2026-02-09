"""Posterization result class and related functionality.

This module provides the Posterization class, which encapsulates the result of
posterizing an image. It includes methods for accessing layer data, colors, SVG
elements, and writing SVG files.

:author: Shay Hill
:created: 2025-02-06
"""

from __future__ import annotations

import dataclasses
import string
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import numpy as np
import svg_ultralight as su
from basic_colormath import rgb_to_hex
from numpy import typing as npt

from posterize.image_processing import layer_to_svgd
from posterize.layers import merge_layers
from posterize.stem_creator import stemize

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    from lxml.etree import (
        _Element as EtreeElement,  # pyright: ignore[reportPrivateUsage]
    )

# a (r, c) array of palette indices in (0,<= 512)
Indices: TypeAlias = Annotated[npt.NDArray[np.intp], "(r, c)"]

# a (n, 3) array of color vectors
Palette: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n, 3)"]

# a (m, n) array. Each row contains a value for one color index. The values will be
# either a single color index or -1 for transparent. Where a color index is present,
# that color will replace colors on lower strate. -1 is transparent, allowing lower
# strata to show through.
Pstrata: TypeAlias = Annotated[npt.NDArray[np.intp], "(m, n)"]

# a (n, r, m) array of layers, each layer with the same shape as the quantized image.
# These are the strata expanded to the same shape as the quantized color indice.
Layers: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, r, c)"]

# a (r, c, 3) array of RGB values
Pixels: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r, c, 3)"]

# 64-char alphabet for compress_int_string (2 chars encode 0-999)
_COMPRESS_ALPHABET = (
    string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_"
)


def compress_int_string(s: str) -> str:
    """Compress a string of digits (length a multiple of 3) into a shorter string.

    Each group of 3 digits is an integer 0-999, encoded in a 64-char alphabet
    (2 chars per group), so 3 digits become 2 chars.

    :param s: string of digit characters; len(s) must be a multiple of 3
    :return: compressed string
    """
    if len(s) % 3 != 0:
        msg = "String length must be a multiple of 3"
        raise ValueError(msg)
    out: list[str] = []
    for i in range(0, len(s), 3):
        n = int(s[i : i + 3])
        if n < 0 or n > 999:
            msg = "Each 3-digit group must be in 0-999"
            raise ValueError(msg)
        out.append(_COMPRESS_ALPHABET[n // 64] + _COMPRESS_ALPHABET[n % 64])
    return "".join(out)


def _get_layer_color_index(layer: npt.NDArray[np.intp]) -> int:
    """Extract the one non -1 color index from an array layer."""
    return int(next(x for x in layer.flatten() if x != -1))


def _get_layer_color(palette: Palette, layer: npt.NDArray[np.intp]) -> str:
    """Extract the one non -1 color index from an array layer."""
    color_index = _get_layer_color_index(layer)
    return rgb_to_hex(palette[color_index])


@dataclasses.dataclass
class Posterization:
    """Result of posterizing an image.

    Holds the quantized palette and indices, proximity matrix and weights from the
    target, the layer strata chosen by the approximation, and the weights used. Also
    provides derived data (layers, colors, SVG path data) and methods to write SVG.

    :param palette: (n, 3) array of color vectors
    :param indices: (r, c) array of palette indices for each pixel
    :param pmatrix: (n, n) proximity matrix between palette colors (from target)
    :param weights: (n,) weight per palette color (from target)
    :param pstrata: (m, n) array of m strata; each row has one color index and -1 for
        transparent. Top strata replace lower ones where they apply
    :param savings_weight: weight used for the savings metric vs average savings
    :param vibrant_weight: weight used for the vibrance metric
    :param source_stem: file stem of the source image (default 'from_array' for
        in-memory sources)
    """

    palette: Annotated[npt.NDArray[np.uint8], "(n, 3)"]
    indices: Annotated[npt.NDArray[np.intp], "(r, c)"]
    pmatrix: Annotated[npt.NDArray[np.float64], "(n, n)"]
    weights: Annotated[npt.NDArray[np.float64], "(n,)"]
    strata: Annotated[npt.NDArray[np.intp], "(m, n)"]
    savings_weight: float
    vibrant_weight: float
    source_stem: str = "from_array"

    def __init__(
        self,
        palette: Annotated[npt.NDArray[np.uint8], "(n, 3)"] | npt.ArrayLike,
        indices: Annotated[npt.NDArray[np.intp], "(r, c)"] | npt.ArrayLike,
        pmatrix: Annotated[npt.NDArray[np.float64], "(n, n)"] | npt.ArrayLike,
        weights: Annotated[npt.NDArray[np.float64], "(n,)"] | npt.ArrayLike,
        pstrata: Annotated[npt.NDArray[np.intp], "(m, n)"] | npt.ArrayLike,
        savings_weight: float,
        vibrant_weight: float,
        source_stem: str = "from_array",
        *,
        accumulated_svgds: list[str] | None = None,
    ) -> None:
        """Initialize the Posterization.

        :param palette: (n, 3) array of color vectors
        :param indices: (r, c) array with palette indices from the quantized image
        :param pmatrix: (n, n) proximity matrix between palette colors (from target)
        :param weights: (n,) weight per palette color (from target)
        :param pstrata: (m, n) array of n layers, each containing a value (color
            index) and -1 for transparent
        :param savings_weight: weight used for the savings metric vs average savings
        :param vibrant_weight: weight used for the vibrance metric
        :param source_stem: file stem of the source image (default 'from_array')
        """
        self.palette = np.asarray(palette, dtype=np.uint8)
        self.indices = np.asarray(indices, dtype=np.intp)
        self.pmatrix = np.asarray(pmatrix, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.strata = np.asarray(pstrata, dtype=np.intp)
        self.savings_weight = savings_weight
        self.vibrant_weight = vibrant_weight
        self.source_stem = source_stem

        self.bbox = su.BoundingBox(0, 0, self.indices.shape[1], self.indices.shape[0])
        self.color_indices = [_get_layer_color_index(x) for x in self.strata]
        self.colors = [rgb_to_hex(self.palette[x]) for x in self.color_indices]
        self.layers = [s[self.indices] for s in self.strata]
        self.full_pixels = self.palette[self.indices]
        self.part_pixels = merge_layers(*self.layers)
        # lazy svgds. Pass to keep these for another instance with more layers.
        self.accumulated_svgds = accumulated_svgds or []

    @property
    def svgds(self) -> list[str]:
        """SVG path data string for each layer."""
        while len(self.accumulated_svgds) < len(self.layers):
            self.accumulated_svgds.append(
                layer_to_svgd(self.layers[len(self.accumulated_svgds)])
            )
        return self.accumulated_svgds

    @property
    def stem(self) -> str:
        """Filename stem from source_stem, color indices, weights, and image size."""
        cols = compress_int_string("".join(f"{x:03d}" for x in self.color_indices))
        max_dim = max(self.indices.shape)
        return stemize(
            self.source_stem,
            cols,
            self.savings_weight,
            self.vibrant_weight,
            max_dim,
        )

    def new_elements(self, num: int | None = None) -> Iterator[EtreeElement]:
        """Create new SVG path elements for each layer.

        :param num: number of layers to create (default is all)
        :yield: SVG path elements
        """
        for color, svgd in zip(self.colors[:num], self.svgds[:num], strict=True):
            yield su.new_element("path", d=svgd, fill=color)

    def new_elem(self, num: int | None = None) -> EtreeElement:
        """Create a new group element with the given number of layers."""
        tmat = f"matrix(1 0 0 -1 0 {self.bbox.height})"
        elem = su.new_element("g", transform=tmat)
        for path in self.new_elements(num):
            elem.append(path)
        return elem

    def new_blem(self, num_cols: int | None = None) -> su.BoundElement:
        """Get a bound element from the group element."""
        elem = self.new_elem(num_cols)
        return su.BoundElement(elem, self.bbox)

    def new_root(self, num_cols: int | None = None) -> EtreeElement:
        """Get the SVG root element."""
        return su.new_svg_root_around_bounds(self.new_blem(num_cols))

    def write_svg(self, path: str | os.PathLike[str], num: int | None = None) -> Path:
        """Write the posterization to an SVG file.

        :param path: path to the output SVG file
        :param num: optionally create the SVG with only the first `num` colors
        :return: path to the output SVG file
        """
        root = self.new_root(num)
        return su.write_svg(Path(path), root)
