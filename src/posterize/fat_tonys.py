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


# _BG_COLOR = "#ac58fc"  # Sebastian's purple
_BG_COLOR = "#b266b2"
_PINSTRIPE_SCALE = 0.015
_STROKE_SCALE = 0.01
_PINSTRIPE_COLOR = _hex_interp(_BG_COLOR, "#ff0000", 0.6)
_STROKE_COLOR = _hex_interp(_BG_COLOR, "#ffffff", 0.4)

_COLOR_STEPS = 5

# ==============================================================================
#
# re-use these time sequences (lux progressions) for any image
#
# ==============================================================================

# re-use these for any image
times = list(np.linspace(0, 1, _COLOR_STEPS, endpoint=False))
time_sequences = [dist.linear(times)]

for s in np.linspace(0.25, 1, 3):
    time_sequences.append(dist.push_f(times, s))
    time_sequences.append(dist.push_l(times, s))
    time_sequences.append(dist.push_i(times, s))
    time_sequences.append(dist.push_o(times, s))




def _get_output_name(input_: Union[Path, str], infix: str) -> Path:
    """Create a custom output name so the input name can be reused."""
    input_ = Path(input_)
    output = input_.parent / f"rendered_{input_.stem}_{infix}.svg"
    return output


def try_lots(input_: Union[Path, str]):
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
            _BG_COLOR,
            (_PINSTRIPE_COLOR, _STROKE_COLOR),
            (pinstripe_width, stroke_width),
        )


try_lots(BINARIES / "nnt_nobg.png")
