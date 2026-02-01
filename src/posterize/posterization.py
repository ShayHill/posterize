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
from typing import TYPE_CHECKING, Annotated, TypeAlias, cast

import numpy as np
import svg_ultralight as su
from basic_colormath import rgb_to_hex
from numpy import typing as npt

from posterize.image_processing import layer_to_svgd

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

    from lxml.etree import (
        _Element as EtreeElement,  # pyright: ignore[reportPrivateUsage]
    )

_IntA: TypeAlias = npt.NDArray[np.intp]


def _expand_layers(
    quantized_image: Annotated[npt.NDArray[np.intp], "(r, c)"],
    d1_layers: Annotated[npt.NDArray[np.intp], "(n, 512)"],
) -> Annotated[npt.NDArray[np.intp], "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.

    Convert the (usually (512,)) layers of an ImageApproximation to the (n, r, c)
    layers required by draw_posterized_image.
    """
    d1_layers_ = cast("Iterable[npt.NDArray[np.intp]]", d1_layers)
    return np.array([x[quantized_image] for x in d1_layers_])


@dataclasses.dataclass
class Posterization:
    """Result of posterizing an image. Serializable.

    :param indices: (r, c) array with palette indices from the quantized image
    :param palette: (512, 3) array of color vectors
    :param layers: (n, 512) array of n layers, each containing a value (color index)
        and -1 for transparent
    :param expanded_layers: (n, r, c) array of layers expanded to the size of the
        quantized image
    """

    # fields are lists for json serialization
    indices: list[list[int]]
    palette: list[list[int]]
    layers: list[list[int]]

    def __init__(
        self,
        indices: Annotated[npt.NDArray[np.intp], "(r, c)"] | list[list[int]],
        palette: Annotated[npt.NDArray[np.uint8], "(512, 3)"] | list[list[int]],
        layers: Annotated[npt.NDArray[np.intp], "(n, 512)"] | list[list[int]],
    ) -> None:
        """Initialize the Posterization.

        :param indices: (r, c) array with palette indices from the quantized image
        :param palette: (512, 3) array of color vectors
        :param layers: (n, 512) array of n layers, each containing a value (color index)
            and -1 for transparent

        __init__ will only be passed numpy arrays in this library. The list types are to
        restore a serialized Posterization object.
        """
        indices = np.asarray(indices, dtype=np.intp)
        palette = np.asarray(palette, dtype=np.uint8)
        layers = np.asarray(layers, dtype=np.intp)
        self.indices = indices.tolist()
        self.palette = palette.tolist()
        self.layers = layers.tolist()

        self._color_indices = tuple([next(x for x in y if x != -1) for y in layers])
        self._expanded_layers = _expand_layers(indices, layers)
        self._colors = [rgb_to_hex(palette[x]) for x in self._color_indices]
        self.bbox = su.BoundingBox(0, 0, indices.shape[1], indices.shape[0])
        self._svgds = [layer_to_svgd(x) for x in self._expanded_layers]

    def get_layers(self, num_cols: int | None = None) -> list[_IntA]:
        """Get the layers for the given number of colors."""
        return list(self._expanded_layers[:num_cols])

    def get_pixels(self, num_cols: int | None = None) -> _IntA:
        """Get the color index for each pixel."""
        return merge_layers(*self._expanded_layers[:num_cols])

    def get_colors(self, num_cols: int | None = None) -> list[str]:
        """Get the color for each layer."""
        return self._colors[:num_cols]

    def get_counts(self, num_cols: int | None = None) -> list[int]:
        """Get the number of pixels for each color."""
        bincount = np.bincount(self.get_pixels(num_cols).flatten())
        return [int(bincount[x]) for x in self._color_indices[:num_cols]]

    def get_svgds(self, num_cols: int | None = None) -> list[str]:
        """Get the SVG data strings for each layer."""
        return self._svgds[:num_cols]

    def new_elements(self, num_cols: int | None = None) -> Iterator[EtreeElement]:
        """Create new SVG path elements for each layer."""
        colors = self.get_colors(num_cols)
        svgds = self.get_svgds(num_cols)
        for color, svgd in zip(colors, svgds, strict=True):
            if not svgd:
                continue
            yield su.new_element("path", d=svgd, fill=color)

    def new_elem(self, num_cols: int | None = None) -> EtreeElement:
        """Create a new group element with the given number of layers."""
        tmat = f"matrix(1 0 0 -1 0 {self.bbox.height})"
        elem = su.new_element("g", transform=tmat)
        for path in self.new_elements(num_cols):
            elem.append(path)
        return elem

    def new_blem(self, num_cols: int | None = None) -> su.BoundElement:
        """Get a bound element from the group element."""
        elem = self.new_elem(num_cols)
        return su.BoundElement(elem, self.bbox)

    def new_root(self, num_cols: int | None = None) -> EtreeElement:
        """Get the SVG root element."""
        return su.new_svg_root_around_bounds(self.new_blem(num_cols))

    def write_svg(
        self, path: str | os.PathLike[str], num_cols: int | None = None
    ) -> Path:
        """Write the posterization to an SVG file.

        :param path: path to the output SVG file
        :param num_cols: optionally create the SVG with only the first `num_cols` colors
        :return: path to the output SVG file
        """
        root = self.new_root(num_cols)
        return su.write_svg(Path(path), root)
