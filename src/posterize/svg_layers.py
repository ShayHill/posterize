"""Create one svg layer with potrace.

Define a class that creates svg layers (group elements) with a given lux
(illumination from [0, 1]). A special case, lux == 0, returns an svg path around all
opaque pixels.

This requires writing temporary bmp and svg files to disk in a TemporaryDirectory.

## Usage

    1. create an instance from a picture
    2. instance(lux: float) to get an svg `<g>` element with smaller dark areas as
       lux increases.

These layers can be layered over each other, lower lux to higher, to create a picture.

:author: Shay Hill
:created: 2023-07-06
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import numpy.typing as npt
from lxml import etree

from posterize import image_arrays as ia, paths
from posterize.constants import DEFAULT_MIN_SPECKLE_SIZE_SCALAR

if TYPE_CHECKING:
    from types import TracebackType

    from lxml.etree import _Element as EtreeElement  # type: ignore

# a pixel array of any mode (RGB, RGBA, L, etc.)
_Pixels = Annotated[npt.NDArray[np.uint8], (-1, -1, -1)]


class SvgLayers:
    """Create necessary temp files then generate SVGs at different lux.

    Use with .close() or context manager to clean up temp files.
    """

    def __init__(
        self, input_filename: str | Path, despeckle: float | None = None
    ) -> None:
        """Open a temporary directory and write temporary bmp files.

        :param input_filename: source filename
        :param despeckle: (0, 1) override default minimum speckle size as a fraction
            of the minimum image dimension (i.e., higher means less speckling).
        """
        if despeckle is None:
            despeckle = DEFAULT_MIN_SPECKLE_SIZE_SCALAR

        self._tmpdir_object = tempfile.TemporaryDirectory()
        self._tmpdir_path = Path(self._tmpdir_object.name)
        self._stem = Path(input_filename).stem

        pixels = ia.get_image_pixels(input_filename)
        self._monochrome = self._write_monochrome(pixels)
        self._silhouette = self._write_silhouette(pixels)

        self.height, self.width = pixels.shape[:2]
        self._tsize = min(self.width, self.height) * despeckle

    def _write_monochrome(self, pixels: _Pixels) -> Path:
        """Create a temporart monochrome bitmap for potrace.

        :param pixels: image pixels
        :return: path to temporary monochrome bitmap
        """
        infix = paths.TempBmpInfix.MONOCHROME
        bmp_path = self._tmpdir_path / paths.get_temp_bmp_filename(self._stem, infix)
        ia.write_monochrome_bmp(bmp_path, pixels)
        return bmp_path

    def _write_silhouette(self, pixels: _Pixels) -> Path:
        """Create a temporart silhouette bitmap for potrace.

        :param pixels: image pixels
        :return: path to temporary silhouette bitmap
        """
        infix = paths.TempBmpInfix.SILHOUETTE
        bmp_path = self._tmpdir_path / paths.get_temp_bmp_filename(self._stem, infix)
        ia.write_silhouette_bmp(bmp_path, pixels)
        return bmp_path

    def close(self) -> None:
        """Close the temporary directory."""
        self._tmpdir_object.cleanup()

    def __enter__(self) -> SvgLayers:
        """Do nothing. Temp directory will open itself.

        :return: self
        """
        return self

    def __exit__(
        self,
        exc_type: None | type[Exception],
        exc_value: None | Exception,
        exc_traceback: None | TracebackType,
    ) -> None:
        """Close the temporary directory."""
        self.close()

    def _write_svg(self, lux: float) -> Path:
        """Create an svg for a given illumination.

        :param lux: illumination level
        :return: path to the output svg
        """
        svg_path = self._tmpdir_path / paths.get_temp_svg_filename(self._stem, lux)
        if math.isclose(lux, 0):
            bitmap = self._silhouette
            blacklevel = 0.5
        else:
            bitmap = self._monochrome
            blacklevel = 1 - lux
        # fmt: off
        command = [
            str(paths.POTRACE),
            str(bitmap),
            "-o", str(svg_path),
            "-k", str(blacklevel),
            "-u", "1",  # do not scale svg (points correspond to pixels array)
            "--flat",  # all paths combined in one element
            "-t", str(self._tsize),  # remove speckles
            "-b", "svg",  # output format
        ]
        # fmt: on
        _ = subprocess.run(command, check=True)
        return svg_path

    def __call__(self, lux: float) -> EtreeElement:
        """Get the svg group for a given illumination.

        :param lux: output svg paths will wrap pixels darker than
            (1 - lux).
        :return: An svg group element. The group element contains path elements. Each
            path element surrounds a black area in the image.

        Output svgs from potrace look like this:

        ```xml
        <element tree>
            <root>
                <metadata></metadata>
                <g>
                    <path></path>
                </g>
            </root>
        </element tree>
        ```
        """
        svg_path = self._write_svg(lux)
        root = etree.parse(str(svg_path)).getroot()
        return root[1]
