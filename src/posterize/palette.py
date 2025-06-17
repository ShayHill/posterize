"""Create palettes from colors identified when posterizing an image.

:author: Shay Hill
:created: 2025-02-11
"""

import itertools as it
from pathlib import Path
from typing import Annotated, Iterable, Iterator, TypeAlias

import numpy as np
from basic_colormath import get_delta_e
from cluster_colors import SuperclusterBase
from numpy import typing as npt
from palette_image.color_block_ops import sliver_color_blocks
from palette_image.svg_display import write_palette

from posterize import paths
from posterize.image_processing import draw_approximation
from posterize.main import posterize
from posterize.paths import WORKING

_RGB: TypeAlias = Annotated[npt.NDArray[np.uint8], (3,)]

_WHITE = (255, 255, 255)

PALETTES = paths.WORKING / "palettes"
PALETTES.mkdir(exist_ok=True)

centers = {
    "J Sultan Ali - Fisher Women": (0.5, 0.25),
    "J Sultan Ali - Toga": (0.5, 0.25),
}

def _percentage_infix(float_: float) -> str:
    """Get a string to use in the filename for a percentage."""
    return f"{int(float_*100):02}"


def stemize(*args: Path | float | int | str | None) -> Iterator[str]:
    """Convert args to strings and filter out empty strings."""
    if not args:
        return
    arg, *tail = args
    if arg is None:
        pass
    elif isinstance(arg, str):
        yield arg
    elif isinstance(arg, Path):
        yield arg.stem
    elif isinstance(arg, float):
        assert 0 <= arg <= 1
        yield _percentage_infix(arg)
    else:
        assert isinstance(arg, int)
        yield f"{arg:03d}"
    yield from stemize(*tail)

def _delta_e_from_white(rgb: _RGB) -> float:
    """Calculate the delta E from white for a given RGB color."""
    r, g, b = rgb
    return get_delta_e((r, g, b), _WHITE)


# class SumSupercluster(SuperclusterBase):
#     """A SuperclusterBase that uses divisive clustering."""

#     quality_metric = "max_error"
#     quality_centroid = "weighted_medoid"
#     assignment_centroid = "weighted_medoid"
#     clustering_method = "divisive"


def posterize_to_n_colors(
    image_path: Path,
    num_cols: int,
    net_cols: int = 6,
    *,
    savings_weight: float | None = None,
    vibrant_weight: float | None = None,
) -> list[int] | None:

    state = posterize(
        image_path,
        num_cols,
        savings_weight=savings_weight,
        vibrant_weight=vibrant_weight,
    )
    stem = "-".join(
        stemize(
            image_path, num_cols, net_cols, state.savings_weight, state.vibrant_weight
        )
    )
    print(f"posterizing {stem}")

    colors = state.target.palette[state.layer_colors]
    tuples = [tuple(x) for x in colors]
    colors = [x for x in colors if _delta_e_from_white(x) > 16][:net_cols]
    reqd_no = tuples.index(tuple(colors[-1])) + 1
    if len(colors) < net_cols:
        print(f"--- discarding {stem}")
        return posterize_to_n_colors(
            image_path,
            num_cols + 1,
            net_cols,
            savings_weight=savings_weight,
            vibrant_weight=vibrant_weight,
        )

    
    output_path = WORKING / f"{stem}.svg"
    draw_approximation(output_path, state, reqd_no)

    dist = [1.0] * net_cols

    color_blocks = sliver_color_blocks(colors, dist)
    output_name = PALETTES / f"{stem}.svg"
    center = centers.get(image_path.stem)
    print(f" ooooooooooooooooooooooooooooooo    {center=}")
    write_palette(image_path, color_blocks, output_name, center=center)
    return


s_ws = [0.5, 0.25]
v_ws = [0.0, 0.5]

if __name__ == "__main__":
    pics = [
        "Sci-Fi - Outland.jpg",
    ]
    pics = [x.name for x in paths._PROJECT.glob("tests/resources/*.webp")]
    pics += [x.name for x in paths._PROJECT.glob("tests/resources/*.jpg")]
    pics += [x.name for x in paths._PROJECT.glob("tests/resources/*.png")]
    # pics = ["bronson.jpg"]
    # for pic in pics:
    #     print(pic)
    for pic in pics:
        image_path = paths._PROJECT / f"tests/resources/{pic}"
        if not image_path.exists():
            print(f"skipping {image_path}")
            continue
        print(f"processing {image_path.name}")
        try:
            seen: set[tuple[int, ...]] = set()
            for sw, vw in it.product(s_ws, v_ws):
                _ = posterize_to_n_colors(
                    image_path,
                    num_cols=12,
                    net_cols=12,
                    savings_weight=sw,
                    vibrant_weight=vw,
                )
        except Exception as e:
            raise e

    print("done")

# industry, perseverance, and frugality # make fortune yield - benjamin franklin
# strength, well being, and health
# no man is your enemy. no man is your friend. every man is your teacher. - florence
# scovel hinn
