"""Create Fat Tony's website banners.

:author: Shay Hill
:created: 2023-08-01
"""

from pathlib import Path

import numpy as np
from basic_colormath import hex_to_rgb, rgb_to_hex
from PIL import Image
from typing import Iterable, Sequence
import itertools as it

import posterize.time_distortions as dist
from posterize.main import posterize_with_outline
from posterize.paths import BINARIES
from posterize.svg_layers import get_quantizer

_SOURCE = BINARIES / "nnt_nobg.png"


_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")


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


_BG_COLOR = "#ac58fc"  # Sebastian's purple
# _BG_COLOR = "#b266b2"
_PINSTRIPE_SCALE = 0.015 * 2
_STROKE_SCALE = 0.01 * 2
_PINSTRIPE_COLOR = "#ff0000"
_STROKE_COLOR = "#ffbb55"

_COLOR_STEPS = 5

# ==============================================================================
#
# re-use these time sequences (lux progressions) for any image
#
# ==============================================================================

# re-use these for any image


def _get_distorted_time_sequences(
    path_to_image: str | Path,
    distortion_strengths: Iterable[float],
) -> list[list[float]]:
    """Get a list of time sequences with different distortions.
    
    :param path_to_image: path to an image file
    :param distortion_strengths: [0, 1] how much to distort the time sequences
    :return: list of time sequences.
    """
    get_quantile = get_quantizer(_SOURCE)
    times = list(np.linspace(0, 1, _COLOR_STEPS, endpoint=False))
    q_times = [get_quantile(x) for x in times]

    base_time_sequences = [dist.linear(times), dist.linear(q_times)]
    distortions = (dist.push_f, dist.push_o, dist.push_i, dist.push_l)

    time_sequences = base_time_sequences[:]
    for d, s, ts in it.product(distortions, distortion_strengths, base_time_sequences):
        time_sequences.append(d(ts, s))
    return time_sequences


time_sequences = _get_distorted_time_sequences(_SOURCE, np.linspace(0.25, 1, 3))


def _get_output_name(input_: Path | str, infix: str) -> Path:
    """Create a custom output name so the input name can be reused.

    :param input_: The input file name.
    :param infix: The infix to add to the output name.
    :return: path to output_infix.svg.
    """
    input_ = Path(input_)
    output = input_.parent / f"rendered_{input_.stem}_{infix}.svg"
    return output


def try_lots(input_: Path | str):
    image = Image.open(input_)
    pinstripe_width = image.height * _PINSTRIPE_SCALE
    stroke_width = image.height * _STROKE_SCALE
    cols = [
        _hex_interp("#ffffff", "#000000", t) for t in np.linspace(0, 1, _COLOR_STEPS)
    ]
    for i, luxs in enumerate(time_sequences):
        print(luxs)
        posterize_with_outline(
            input_,
            _get_output_name(input_, str(i)),
            luxs,
            cols,
            background=_BG_COLOR,
            inkscape=_INKSCAPE,
            strokes=[
                {
                    "stroke": _PINSTRIPE_COLOR,
                    "stroke-width": str(pinstripe_width),
                    "stroke-opacity": "0.7",
                },
                {
                    "stroke": _STROKE_COLOR,
                    "stroke-width": stroke_width,
                    "stroke-opacity": "0.7",
                },
            ],
        )


try_lots(BINARIES / "nnt_nobg.png")
