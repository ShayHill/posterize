"""Run the posterization functions to create output.

:author: Shay Hill
:created: 2025-11-16
"""

import os
import time
from pathlib import Path

from posterize import posterize
from posterize.posterization import dump_posterization, load_posterization

INKSCAPE = str(Path(r"C:\Program Files\Inkscape\bin\inkscape"))


PROJECT = Path(__file__).parents[2]
RESOURCES = PROJECT / "resources"


def _dennis_ritchie(svg: str | os.PathLike[str]) -> Path:
    posterization = posterize(RESOURCES / "dennis_ritchie.png", 5)
    print(posterization.get_counts())
    return posterization.write_svg(svg)


def _marbles(svg: str | os.PathLike[str]) -> Path:
    posterization = posterize(RESOURCES / "marbles2.png", 5)
    print(posterization.get_counts())
    dump_posterization(posterization, "marbles.json")
    posterization = load_posterization("marbles.json")
    print(posterization.get_counts())
    return posterization.write_svg(svg)


if __name__ == "__main__":
    start_time = time.time()
    # _ = _dennis_ritchie("dennis_ritchie.svg")
    _ = _marbles("marbles.svg")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")  # noqa: T201
