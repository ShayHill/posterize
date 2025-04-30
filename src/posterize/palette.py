"""Create palettes from colors identified when posterizing an image.

:author: Shay Hill
:created: 2025-02-11
"""

import itertools as it
from pathlib import Path

from cluster_colors import SuperclusterBase
from lxml.etree import _Element as EtreeElement  # type: ignore
from palette_image.color_block_ops import sliver_color_blocks
from palette_image.svg_display import write_palette

from posterize import paths
from posterize.iterative_main import (
    draw_approximation,
    posterize,
    stemize,
)


INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")

PALETTES = paths.WORKING / "palettes"
PALETTES.mkdir(exist_ok=True)


class SumSupercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "max_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"


def posterize_to_n_colors(
    image_path: Path,
    num_cols: int,
    net_cols: int = 6,
    *,
    savings_weight: float | None = None,
    vibrant_weight: float | None = None,
) -> list[int] | None:


    state = posterize(
        image_path, 6, savings_weight=savings_weight, vibrant_weight=vibrant_weight
    )
    stem = "-".join(
        stemize(
            image_path, num_cols, net_cols, state.savings_weight, state.vibrant_weight
        )
    )
    print(f"posterizing {stem}")

    draw_approximation(image_path, state, 6, stem)

    colors = state.layer_colors

    vectors = state.target.palette[colors]

    dist = [1.0] * len(colors)

    color_blocks = sliver_color_blocks(vectors, dist)
    output_name = PALETTES / f"{stem}.svg"
    write_palette(image_path, color_blocks, output_name)
    return


s_ws = [0.5, 0.25]
v_ws = [0.0, 0.5]

if __name__ == "__main__":
    pics = [
        "Sci-Fi - Outland.jpg",
    ]
    pics = [x.name for x in paths.PROJECT.glob("tests/resources/*.webp")]
    pics += [x.name for x in paths.PROJECT.glob("tests/resources/*.jpg")]
    # pics = ["bronson.jpg"]
    # for pic in pics:
    #     print(pic)
    for pic in pics:

        image_path = paths.PROJECT / f"tests/resources/{pic}"
        if not image_path.exists():
            print(f"skipping {image_path}")
            continue
        print(f"processing {image_path.name}")
        try:
            seen: set[tuple[int, ...]] = set()
            for sw, vw in it.product(s_ws, v_ws):
                _ = posterize_to_n_colors(
                    image_path,
                    num_cols=6,
                    net_cols=6,
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
