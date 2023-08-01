"""Create Fat Tony's website banners.

:author: Shay Hill
:created: 2023-08-01
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union

from posterize.paths import BINARIES
from basic_colormath import hex_to_rgb, rgb_to_hex
import posterize.time_distortions as dist

from posterize.main import posterize_with_outline

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

_BG_COLOR = "#ac58fc"
_BG_COLOR = "#b266b2"
_PINSTRIPE_SCALE = 0.015
_STROKE_SCALE = 0.01
_PINSTRIPE_COLOR = _hex_interp(_BG_COLOR, "#ff0000", 0.6)
_STROKE_COLOR = _hex_interp(_BG_COLOR, "#ffffff", 0.4)

_COLOR_STEPS = 5


# re-use these for any image
times = list(np.linspace(0, 1, _COLOR_STEPS))
time_sequences = [dist.interpolate_floats(times, times, 1)]
for s in np.linspace(0.25, 1, 3):
    time_sequences.append(dist.q1_time(times, s))
    time_sequences.append(dist.q3_time(times, s))
    time_sequences.append(dist.cos_time(times, s))
    time_sequences.append(dist.sin_time(times, s))
for p in np.linspace(1.25, 2, 3):
    time_sequences.append(dist.pow_time(times, p))
    time_sequences.append(dist.pow_time(times, 1 / p))




def linear_interpolation(
    input: Union[Path, str],
    output: Union[Path, str],
    num_layers: int,
    black: str = "#000000",
    white: str = "#ffffff",
    stroke_shade: str = "#ff0000",
) -> None:
    """A default posterized effect.

    :param input: The input png file.
    :param output: The output svg file.
    :param num_layers: The number of layers to use.
    :param black: The color to use for the darkest layer.
    :param white: The color to use for the lightest layer.
    :param stroke_shade: The color to mix with the background to create the outline.

    Will use a medium color for the background.
    """
    power = 1
    luxs = np.linspace(0, 0.8**power, num_layers)
    luxs = luxs ** (1 / power)
    background = _hex_interp(white, black, 0.5)

    black = "#000000"
    white = "#eeeeee"
    cols = [_hex_interp(white, black, lux) for lux in luxs]
    stroke = _hex_interp(background, stroke_shade, 0.5)
    stroke_width = 5
    posterize_with_outline(input, output, luxs, cols, background, stroke, stroke_width)


def _get_output_name(input_: Union[Path, str], infix: str) -> Path:
    """Create a custom output name so the input name can be reused."""
    input_ = Path(input_)
    output = input_.parent / f"rendered_{input_.stem}_{infix}.svg"
    return output


def try_lots(input_: Union[Path, str]):
    image = Image.open(input_)
    pinstripe_width = image.height * _PINSTRIPE_SCALE
    stroke_width = image.height * _STROKE_SCALE
    cols = (
        [_hex_interp("#ffffff", "#000000", t) for t in np.linspace(0, 1, _COLOR_STEPS)]
    )
    for i, luxs in enumerate(time_sequences):
        print(luxs)
        posterize_with_outline(
            input_,
            _get_output_name(input_, str(i)),
            luxs,
            cols,
            _BG_COLOR,
            (_PINSTRIPE_COLOR, _STROKE_COLOR),
            (pinstripe_width, stroke_width),
        )


try_lots(BINARIES / "nnt_nobg.png")
