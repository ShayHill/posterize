"""Create one svg layer with potrace.

Define a class that creates svg layers (group elements) with a given illumination. A
special case, illumination == 0, returns an svg path around all opaque pixels.

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

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from lxml import etree

from posterize import image_arrays as ia
from posterize import paths
from posterize.constants import DEFAULT_MIN_SPECKLE_SIZE_SCALAR

if TYPE_CHECKING:
    from lxml.etree import _Element as EtreeElement  # type: ignore


class SvgLayers:
    """Create necessary temp files then generate SVGs at different illumination
    levels.

    Use with .close() or context manager to clean up temp files.
    """

    def __init__(
        self,
        input_filename: Union[str, Path],
        despeckle: Optional[float] = None,
    ) -> None:
        """Open a temporary directory and write temporary bmp files.

        :param input_filename: source filename
        :param despeckle: (0, 1) override default minimum speckle size as a fraction
            of the minimum image dimension (i.e., higher means less speckling).
        """
        if despeckle is None:
            despeckle = DEFAULT_MIN_SPECKLE_SIZE_SCALAR
        self._tmpdir = tempfile.TemporaryDirectory()
        self._stem = Path(input_filename).stem

        pixels = ia.get_image_pixels(input_filename)
        self.height, self.width = pixels.shape[:2]
        self.min_dim = min(self.height, self.width)

        self._tsize = self.min_dim * despeckle

        monochrome = paths.get_temp_bmp_filename(
            input_filename, paths.TempBmpInfix.MONOCHROME
        )
        silhouette = paths.get_temp_bmp_filename(
            input_filename, paths.TempBmpInfix.SILHOUETTE
        )
        self._monochrome = os.path.join(self._tmpdir.name, monochrome)
        self._silhouette = os.path.join(self._tmpdir.name, silhouette)
        ia.write_monochrome_bmp(self._monochrome, pixels)
        ia.write_silhouette_bmp(self._silhouette, pixels)

    def close(self) -> None:
        """Close the temporary directory."""
        self._tmpdir.cleanup()

    def __enter__(self) -> SvgLayers:
        """Do nothing. Temp directory will open itself."""
        return self

    def __exit__(
        self,
        exc_type: Any,  # None | Type[Exception], but py <= 3.9 doesn't like it.
        exc_value: Any,  # None | Exception, but py <= 3.9 doesn't like it.
        exc_traceback: Any,  # None | TracebackType, but py <= 3.9 doesn't like it.
    ):
        self.close()

    def _write_svg(self, illumination: float):
        """Create an svg for a given illumination."""
        svg_filename = paths.get_temp_svg_filename(self._stem, illumination)
        svg_path = os.path.join(self._tmpdir.name, svg_filename)
        if illumination == 0:
            bitmap = self._silhouette
            blacklevel = 0.5
        else:
            bitmap = self._monochrome
            blacklevel = 1 - illumination
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
        _ = subprocess.run(command)
        return svg_path

    def __call__(self, illumination: float) -> EtreeElement:
        """Get the svg group for a given illumination.

        :param illumination: output svg paths will wrap pixels darker than
            (1 - illumination).
        :return: An svg group element. The group element contains path elements. Each
            path element surrounds a black area in the image.

        Output svgs from potrace look like this:

        ```xml
        <element tree>
            <root>
                <metadata></metadata>
                <g></g>
            </root>
        </element tree>
        ```
        """
        svg_path = self._write_svg(illumination)
        root = etree.parse(str(svg_path)).getroot()
        return root[1]
