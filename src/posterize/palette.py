"""Create palettes from colors identified when posterizing an image.

:author: Shay Hill
:created: 2025-02-11
"""

import functools as ft
from pathlib import Path

import numpy as np
import svg_ultralight as su
from basic_colormath import rgb_to_hsv, rgbs_to_hsv
from cluster_colors import Members, SuperclusterBase
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt
from palette_image.color_block_ops import sliver_color_blocks
from palette_image.svg_display import write_palette

from posterize import paths
from posterize.iterative_main import Supercluster, draw_approximation, posterize


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
    ixs: npt.ArrayLike,
    bite_size: float,
    num_cols: int,
    seen: set[tuple[int, ...]] | None = None,
    pick_: list[int | None] | None = None,
    min_dist: float = 16,
) -> list[int] | None:

    print(f"{image_path.stem} {min_dist}")

    state = posterize(image_path, 6, ignore_cache=False)
    # draw_approximation(state, 6, "input_06")
    # draw_approximation(state, 12, "input_12")
    draw_approximation(image_path, state, 6, "input_16")
    # draw_approximation(state, 24, "input_24")


    colors = state.layer_colors
    vectors = state.target.palette[colors]

    return

    for boost_delta_h in range(11):
        boost = boost_delta_h / 10
        vibrance_weighted_delta_e = ft.partial(new_vibrance_weighted_delta_e, boost)
        pmatrix = build_proximity_matrix(vectors, vibrance_weighted_delta_e)
        weights = [
            state.get_state_weight(x) * (max(y) - min(y))
            for x, y in zip(colors, vectors, strict=True)
        ]
        members = Members(vectors, weights=weights, pmatrix=pmatrix)
        supercluster = Supercluster(members)

        heaviest = _get_dominant(supercluster, min_members=4)
        heaviest.set_n(4)

        palette = [x.centroid for x in heaviest.clusters]
        breakpoint()

        def get_contrast(palette_: list[int], color: int) -> float:
            # return min(pmatrix[color, palette_]) * (max(vectors[color]) - min(vectors[color]))
            if color in palette_:
                return 0
            rgb = supercluster.members.vectors[color]
            rgbs = supercluster.members.vectors[palette_]
            hsv = rgb_to_hsv(rgb)
            hsvs = rgbs_to_hsv(rgbs)
            vib = max(rgb) - min(rgb)
            vibs = [max(x) - min(x) for x in rgbs]
            hsp = [x[0] for x in hsvs]
            hsc = [hsv[0]] * len(hsp)
            deltas_h = get_circular_deltas(hsc, hsp) * vibs
            deltas_e = supercluster.members.pmatrix[color, palette_]
            scaled = deltas_e * deltas_h
            return np.mean(scaled) * vib

        while len(palette) < 6:
            free_cols = supercluster.ixs
            next_color = max(free_cols, key=lambda x: get_contrast(palette, x))
            palette.append(next_color)

        dist = [1, 1, 1, 1, 1, 1]

        pvectors = supercluster.members.vectors[palette]
        color_blocks = sliver_color_blocks(pvectors, list(map(float, dist)))
        boost_str = f"{boost:.2f}".replace(".", "_")
        output_name = PALETTES / f"{image_path.stem}_{boost_str}.svg"

        key = (image_path.stem, *tuple(palette))
        if key not in seen:
            write_palette(image_path, color_blocks, output_name)
            su.write_png_from_svg(INKSCAPE, output_name)
        seen.add(key)

    print(f"{len(palette)=}")
    return palette


if __name__ == "__main__":
    pics = [
        # "adidas.jpg",
        # "bird.jpg",
        # "blue.jpg",
        # "broadway.jpg",
        # "bronson.jpg",
        # "cafe_at_arles.jpg",
        # "dolly.jpg",
        # "dutch.jpg",
        # "Ernest - Figs.jpg",
        # "eyes.jpg",
        # "Flâneur - Al Carbon.jpg",
        # "Flâneur - Coffee.jpg",
        # "Flâneur - Japan.jpg",
        # "Flâneur - Japan2.jpg",
        # "Flâneur - Lavenham.jpg",
        # "girl.jpg",
        # "girl_p.jpg",
        # "hotel.jpg",
        # "Johannes Vermeer - The Milkmaid.jpg",
        # "lena.jpg",
        # "lion.jpg",
        # "manet.jpg",
        # "parrot.jpg",
        # "pencils.jpg",
        # "Retrofuturism - One.jpg",
        # "roy_green_car.jpg",
        "Sci-Fi - Outland.jpg",
        # "seb.jpg",
        # "starry_night.jpg",
        # "taleb.jpg",
        # "tilda.jpg",
        # "you_the_living.jpg",
    ]
    pics = [x.name for x in paths.PROJECT.glob("tests/resources/*.jpg")]
    # pics = ["bronson.jpg"]
    # for pic in pics:
    #     print(pic)
    for pic in pics:

        image_path = paths.PROJECT / f"tests/resources/{pic}"
        if not image_path.exists():
            print(f"skipping {image_path}")
            continue
        print(f"processing {image_path}")
        try:
            seen: set[tuple[int, ...]] = set()
            _ = posterize_to_n_colors(
                image_path,
                bite_size=9,
                ixs=(),
                num_cols=6,
                seen=seen,
            )
        except Exception as e:
            raise e

    print("done")

# industry, perseverance, and frugality # make fortune yield - benjamin franklin
# strength, well being, and health
# no man is your enemy. no man is your friend. every man is your teacher. - florence
# scovel shinn
