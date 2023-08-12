"""Create Fat Tony's website banner images.

Writes a lot of candidate images to disk. The first (absolute linear gray
levels) or second (quantized linear gray levels) will usually be the best, but
the remaining are there is can you come across a tough image like a
light-skinned person in a white jacket.

The anticipated use is to tweak the metaparameters in the CONFIGURATION section
and then pass images under `if __name__ == "__main__":`, each of which will use
the same configuration.

Output images will be the same size as input images, so if you need padding for your
pinstripes, you'll have to add that in an editor before you pass the image here. See
image preparation suggesions in README.md.

:author: Shay Hill
:created: 2023-08-01
"""

from __future__ import annotations

import itertools as it
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import hex_to_rgb, rgb_to_hex
from PIL import Image
from svg_ultralight import format_number

import posterize.time_distortions as dist
from posterize.main import posterize_with_outline
from posterize.paths import BINARIES
from posterize.svg_layers import get_quantiler

if TYPE_CHECKING:
    from collections.abc import Iterable


def _hex_interp(hex_a: str, hex_b: str, time: float) -> str:
    """Interpolate between two hex colors.

    :param hex_a: The first hex color.
    :param hex_b: The second hex color.
    :param time: The time to interpolate at. 0.0 returns hex_a, 1.0 returns hex_b.
    :return: The interpolated hex color.
    """
    rgb_a = np.array(hex_to_rgb(hex_a))
    rgb_b = np.array(hex_to_rgb(hex_b))
    r, g, b = np.round(rgb_a + time * (rgb_b - rgb_a))
    return rgb_to_hex((r, g, b))


_PANTONE_123C = "#ffc62d"

# ========================================================================
# CONFIGURATION
# ========================================================================

# This package will produce SVGs always and PNGs if an Inscape binary is provided.
# Some versions on Inkscape will only work if the path is provided *without* the .exe
# extension.
# _INKSCAPE = None
_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")

# Pass a background color to preview the images, but set this to None when you're
# ready for a final render. With background=None, you will get SVGs (and optionally
# PNGs) with a transparent background.
_BG_COLOR = "#ac58fc"  # Sebastian's purple
# _BG_COLOR = None

# Pinstripes around the silhouette. Widest first. Higher layers mask lower, so
# multiple, partially transparent layers will mix with the background, but *not* with
# each other. Caution with stroke-width. I am using the same svg attribute as an
# absolute width, but it is in fact a scalar. `"stroke-width": 0.03` means 3% of the
# image height.
_PINSTRIPES: list[dict[str, str | float]] = [
    {"stroke": "#ff0000", "stroke-width": 0.03, "stroke-opacity": 0.6},
    {"stroke": _PANTONE_123C, "stroke-width": 0.02, "stroke-opacity": 0.9},
]

# How many gray levels to use for the image.
_STEPS = 5

# More distortion strengths will produce more candidate images. The default three
# will give you 26 candidate images.
_DISTORTION_STRENGTHS = [0.25, 0.5, 0.75]

# darkest color to use
_BLACK = _hex_interp("#000000", _PANTONE_123C, 0.05)

# lightest color to use
_WHITE = _hex_interp("#ffffff", _PANTONE_123C, 0.05)

# ========================================================================
# /CONFIGURATION
# ========================================================================


def _get_distorted_time_sequences(
    path_to_image: str | Path, distortion_strengths: Iterable[float]
) -> list[list[float]]:
    """Get a list of time sequences with different distortions.

    :param path_to_image: path to an image file
    :param distortion_strengths: [0, 1] how much to distort the time sequences
    :return: list of time sequences.
    """
    get_quantile = get_quantiler(path_to_image)
    times = list(np.linspace(0, 1, _STEPS, endpoint=False))
    q_times = [get_quantile(x) for x in times]

    base_time_sequences = [dist.linear(times), dist.linear(q_times)]
    distortions = (dist.push_f, dist.push_o, dist.push_i, dist.push_l)

    time_sequences = base_time_sequences[:]
    for d, s, ts in it.product(distortions, distortion_strengths, base_time_sequences):
        time_sequences.append(d(ts, s))
    return time_sequences


def _get_output_name(path_to_image: Path | str, infix: str) -> Path:
    """Create a custom output name so the input name can be reused.

    :param path_to_image: The input file name.
    :param infix: The infix to add to the output name.
    :return: path to output_infix.svg.
    """
    path_to_image = Path(path_to_image)
    return path_to_image.parent / f"rendered_{path_to_image.stem}_{infix}.svg"


def produce_candidate_images(path_to_image: Path | str):
    """Write several candidate SVGs (and optionally PNGs) to disk.

    :param path_to_image: path to an image file
    :effects: writes several SVGs (and optionally PNGs) to whatever folder
    path_to_image was in.

    """
    image = Image.open(path_to_image)

    # scale the pinstripe widths to some ratio of the image height
    pinstripes = _PINSTRIPES[:]
    for pinstripe in (p for p in pinstripes if "stroke-width" in p):
        stroke_width = float(pinstripe["stroke-width"])
        pinstripe["stroke-width"] = format_number(stroke_width * image.height)

    time_sequences = _get_distorted_time_sequences(path_to_image, _DISTORTION_STRENGTHS)
    cols = [_hex_interp(_WHITE, _BLACK, t) for t in np.linspace(0, 1, _STEPS)]

    for i, luxs in enumerate(time_sequences):
        posterize_with_outline(
            path_to_image,
            _get_output_name(path_to_image, str(i)),
            luxs,
            cols,
            inkscape=_INKSCAPE,
            background=_BG_COLOR,
            strokes=pinstripes,
        )


if __name__ == "__main__":
    produce_candidate_images(BINARIES / "nnt_nobg.png")
