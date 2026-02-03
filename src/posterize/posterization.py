"""Posterization result class and related functionality.

This module provides the Posterization class, which encapsulates the result of
posterizing an image. It includes methods for accessing layer data, colors, SVG
elements, and writing SVG files.

:author: Shay Hill
:created: 2025-02-06
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import numpy as np
import svg_ultralight as su
from basic_colormath import rgb_to_hex
from numpy import typing as npt

from posterize.image_processing import layer_to_svgd
from posterize.layers import merge_layers

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


def _get_layer_color(palette: Palette, layer: npt.NDArray[np.intp]) -> str:
    color_index = next(x for x in layer.flatten() if x != -1)
    return rgb_to_hex(palette[color_index])


@dataclasses.dataclass
class Posterization:
    """Result of posterizing an image. Serializable.

    :param indices: (r, c) array with palette indices from the quantized image
    :param palette: (n, 3) array of color vectors
    :param pstrata: (m, n) array of n sequences, each containing one color index and
        -1 for transparent. For each stratum, the one color index will appear over lower
        colors it will replace.
    """

    indices: Annotated[npt.NDArray[np.intp], "(r, c)"]
    palette: Annotated[npt.NDArray[np.uint8], "(n, 3)"]
    pstrata: Annotated[npt.NDArray[np.intp], "(m, n)"]

    def __init__(
        self,
        indices: Annotated[npt.NDArray[np.intp], "(r, c)"] | npt.ArrayLike,
        palette: Annotated[npt.NDArray[np.uint8], "(n, 3)"] | npt.ArrayLike,
        pstrata: Annotated[npt.NDArray[np.intp], "(m, n)"] | npt.ArrayLike,
    ) -> None:
        """Initialize the Posterization.

        :param indices: (r, c) array with palette indices from the quantized image
        :param palette: (512, 3) array of color vectors
        :param pstrata: (n, 512) array of n layers, each containing a value (color
            index) and -1 for transparent
        """
        self.indices = np.asarray(indices, dtype=np.intp)
        self.palette = np.asarray(palette, dtype=np.uint8)
        self.pstrata = np.asarray(pstrata, dtype=np.intp)

        self.bbox = su.BoundingBox(0, 0, self.indices.shape[1], self.indices.shape[0])
        self.layers = [self.pstrata[0][self.indices]]
        self.svgds = [layer_to_svgd(self.layers[0])]
        for stratum in self.pstrata[1:]:
            layer = stratum[self.indices]
            svgd = layer_to_svgd(layer)
            if not svgd:
                continue
            self.layers.append(layer)
            self.svgds.append(svgd)
        self.full_pixels = self.palette[self.indices]
        self.part_pixels = merge_layers(*self.layers)
        self.colors = [_get_layer_color(self.palette, x) for x in self.layers]
        self.counts = [int(np.sum(layer != -1)) for layer in self.layers]

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
