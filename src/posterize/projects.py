"""Run the posterization functions to create output.

:author: Shay Hill
:created: 2025-11-16
"""

import time
from pathlib import Path

import numpy as np
import svg_ultralight as su
from basic_colormath import get_delta_e_matrix_hex, rgb_to_hex

from posterize import posterize
import os

INKSCAPE = str(Path(r"C:\Program Files\Inkscape\bin\inkscape"))


PROJECT = Path(__file__).parents[2]
RESOURCES = PROJECT / "resources"


def _dennis_ritchie(svg: str | os.PathLike[str]) -> Path:
    image_approximation = posterize(RESOURCES / "dennis_ritchie.png", 5)
    return image_approximation.write_svg(svg)


if __name__ == "__main__":
    start_time = time.time()
    _ = _dennis_ritchie("dennis_ritchie.svg")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")  # noqa: T201
