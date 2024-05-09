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

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree
from stacked_quantile import get_stacked_quantile

from posterize import image_arrays as ia, paths
from posterize.constants import DEFAULT_MIN_SPECKLE_SIZE_SCALAR

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from lxml.etree import _Element as EtreeElement  # type: ignore


def get_quantiler(path_to_image: str | Path) -> Callable[[float], float]:
    """Create a function to return a quantile of pixel graylevels in [0, 1].

    :param path_to_image: path to an image file
    :return: a function that takes a float q in [0, 1] and returns a float y in
        [0, 1] where y is greater than or equal to approximately (100*q)% of
        the gray values in the image (scaled to 1/255).
    """
    pixels = ia.get_image_pixels(path_to_image)
    vs, ws = ia.quantize_pixels(pixels)

    def _get_quantile(q: float) -> float:
        """Get the quantile of the image gray levels.

        :param q: quantile [0, 1]
        :return: quantile value ratio [0, 1]

        Return 0 for 0 to preserve special silhouette case for lux == 0.
        """
        if q == 0:
            return 0
        return get_stacked_quantile(vs, ws, q) / 255

    return _get_quantile


class _FilePaths:
    """Get the filename for a temporary bmp or svg file."""

    def __init__(self, path_to_tmpdir: Path, path_to_image: str | Path) -> None:
        """Get the filename for a temporary bmp or svg file.

        :param path_to_tmpdir: the stem of the filename
        :param path_to_input: the infix of the filename
        """
        self._path_to_tmpdir = path_to_tmpdir
        self._stem = Path(path_to_image).stem

    @property
    def tmp_monochrome(self) -> Path:
        """Get the filename for a temporary monochrome bmp.

        :return: filename for a temporary bmp file
        """
        infix = paths.TmpBmpInfix.MONOCHROME
        return self._path_to_tmpdir / paths.get_tmp_bmp_filename(self._stem, infix)

    @property
    def tmp_silhouette(self) -> Path:
        """Get the filename for a temporary silhouette bmp.

        :return: filename for a temporary bmp file
        """
        infix = paths.TmpBmpInfix.SILHOUETTE
        return self._path_to_tmpdir / paths.get_tmp_bmp_filename(self._stem, infix)

    def get_tmp_svg(self, lux: float) -> Path:
        """Get the filename for a temporary svg.

        :param lux: illumination level
        :return: filename for a temporary svg file
        """
        return self._path_to_tmpdir / paths.get_tmp_svg_filename(self._stem, lux)


class SvgLayers:
    """Create necessary temp files then generate SVGs at different lux.

    Use with .close() or context manager to clean up temp files.
    """

    def __init__(
        self, path_to_image: str | Path, despeckle: float | None = None
    ) -> None:
        """Open a temporary directory and write temporary bmp files.

        :param input_filename: source filename
        :param despeckle: (0, 1) override default minimum speckle size as a fraction
            of the minimum image dimension (i.e., higher means less speckling).
        """
        if despeckle is None:
            despeckle = DEFAULT_MIN_SPECKLE_SIZE_SCALAR
        self._tmpdir = tempfile.TemporaryDirectory()

        self._file_paths = _FilePaths(Path(self._tmpdir.name), path_to_image)

        pixels = ia.get_image_pixels(path_to_image)
        ia.write_monochrome_bmp(self._file_paths.tmp_monochrome, pixels)
        ia.write_silhouette_bmp(self._file_paths.tmp_silhouette, pixels)

        self.height, self.width = pixels.shape[:2]
        self._tsize = self.height * despeckle

    def close(self) -> None:
        """Close the temporary directory."""
        self._tmpdir.cleanup()

    def __enter__(self) -> SvgLayers:
        """Do nothing. Temp directory will open itself.

        :return: self
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Close the temporary directory."""
        self.close()

    def _write_svg(self, lux: float) -> Path:
        """Create an svg for a given illumination.

        :param lux: illumination level
        :return: path to the output svg
        """
        svg_path = self._file_paths.get_tmp_svg(lux)
        if lux == 0:
            bitmap = self._file_paths.tmp_silhouette
            blacklevel = 0.5
        else:
            bitmap = self._file_paths.tmp_monochrome
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
