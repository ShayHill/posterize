"""Create palettes from colors identified when posterizing an image.

:author: Shay Hill
:created: 2025-02-11
"""

import logging
import numpy as np
from pathlib import Path
from posterize.iterative_main import Supercluster, posterize, TargetImage, draw_target

from palette_image.svg_display import write_palette
from palette_image.color_block_ops import sliver_color_blocks

import numpy as np
from basic_colormath import (
    rgb_to_hsv,
    get_deltas_e,
    rgbs_to_hsv,
)
from cluster_colors import SuperclusterBase, Members
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt

from posterize import paths

from typing import Any, Callable

logging.basicConfig(level=logging.INFO)


def build_proximity_matrix(
    colors: npt.ArrayLike,
    func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating[Any]]],
) -> npt.NDArray[np.floating[Any]]:
    """Build a proximity matrix from a list of colors.

    :param colors: an array (n, d) of Lab or rgb colors
    :param func: a commutative, vectorized function that calculates the proximity
        between two colors. It is assumed that identical colors have a
        proximity of 0.
    :return: an array (n, n) of proximity values between every pair of colors

    The proximity matrix is symmetric.
    """
    colors = np.asarray(colors)
    n = len(colors)
    rows = np.repeat(colors[:, np.newaxis, :], n, axis=1)
    cols = np.repeat(colors[np.newaxis, :, :], n, axis=0)
    proximity_matrix = np.zeros((n, n))
    ut = np.triu_indices(n, k=1)
    lt = (ut[1], ut[0])
    proximity_matrix[ut] = func(cols[ut], rows[ut])
    proximity_matrix[lt] = proximity_matrix[ut]
    return proximity_matrix


def get_circular_deltas(
    color_a: npt.ArrayLike, color_b: npt.ArrayLike
) -> npt.NDArray[np.floating[Any]]:
    """Get the circular pairwise deltas between two arrays of hues in [0, 360).

    :param color_a: A 1D array of hue values in [0, 360)
    :param color_b: A 1D array of hue values in [0, 360%)
    :return: A lD array of circular pairwise distances between a and b
    """
    deltas_h = np.abs(np.subtract(color_a, color_b))
    return np.minimum(deltas_h, 360 - deltas_h)


def vibrance_weighted_delta_e(color_a: npt.ArrayLike, color_b: npt.ArrayLike) -> float:
    """Get the delta E between two colors weighted by their vibrance.

    :param color_a: (r, g, b) color, TargetImage
    :param color_b: (r, g, b) color
    :return: delta E between color_a and color_b weighted by their vibrance

    The vibrance of a color is the distance between the color and a pure version of
    the color. The delta E is then multiplied by the vibrance of both colors.
    """
    color_a = np.asarray(color_a)
    color_b = np.asarray(color_b)
    deltas_e = get_deltas_e(color_a, color_b)
    vibrancies_a = np.max(color_a, axis=1) - np.min(color_a, axis=1)
    vibrancies_b = np.max(color_b, axis=1) - np.min(color_b, axis=1)
    hsvs_a = rgbs_to_hsv(color_a)
    hsvs_b = rgbs_to_hsv(color_b)
    deltas_h = get_circular_deltas(hsvs_a[:, 0], hsvs_b[:, 0])
    return deltas_e * ((255 * 180) + np.min([vibrancies_a, vibrancies_b], axis=0) * deltas_h)


class SumSupercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "max_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"

def _get_dominant(supercluster: SuperclusterBase, min_members: int = 0) -> Supercluster:
    """Try to extract a cluster with a dominant color."""
    full_weight = sum(x.weight for x in supercluster.clusters)
    supercluster.set_n(2)
    heaviest = max(supercluster.clusters, key=lambda x: x.weight)
    if heaviest.weight / full_weight > 1 / 2 and len(heaviest.ixs) >= min_members:
        supercluster = supercluster.copy(inc_members=heaviest.ixs)
        return _get_dominant(supercluster, min_members)
    return supercluster


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
    bite_size = 24
    while bite_size >= 0:
        target = TargetImage(image_path, bite_size)
        vectors = target.clusters.members.vectors
        # break

        # strip away whites
        # new_ixs = target.clusters.ixs
        # new_ixs = [x for x in new_ixs if min(vectors[x]) < 90]
        # new_ixs = [x for x in new_ixs if max(vectors[x]) > 90]
        # new_ixs = [x for x in new_ixs if rgb_to_hsv(vectors[x])[1] > 40]
        # vs = [tuple(map(int, vectors[x])) for x in new_ixs]
        # new_ixs_array = np.array(new_ixs, dtype=np.int32)
        # target.clusters = target.clusters.copy(inc_members=new_ixs_array)

        target = posterize(image_path, 12, ixs, 16, ignore_cache=False)
        draw_target(target, 6, "input_06")
        draw_target(target, 12, "input_12")
        draw_target(target, 16, "input_16")
        draw_target(target, 24, "input_24")
        break

    target = TargetImage(image_path, bite_size)
    target = posterize(image_path, 12, None, 16, ignore_cache=False)

    colors = [int(max(x)) for x in target._layers]
    vectors = target.clusters.members.vectors[colors]
    pmatrix = build_proximity_matrix(vectors, vibrance_weighted_delta_e)
    weights = [
        target.get_state_weight(x) * (max(y) - min(y))
        for x, y in zip(colors, vectors, strict=True)
    ]
    members = Members(vectors, weights=weights, pmatrix=pmatrix)
    supercluster = Supercluster(members)

    heaviest = _get_dominant(supercluster, min_members=4)
    heaviest.set_n(4)
    aaa = heaviest.get_as_vectors()

    palette = [x.centroid for x in heaviest.clusters]



    def get_contrast(palette_: list[int], color: int) -> float:
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
    output_name = paths.WORKING / f"{image_path.stem}.svg"

    write_palette(image_path, color_blocks, output_name)

    seen.add(tuple(palette))
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
    # pics = [x.name for x in paths.PROJECT.glob("tests/resources/*.jpg")]
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
            pass

    print("done")

# industry, perseverance, and frugality # make fortune yield - benjamin franklin
# strength, well being, and health
# no man is your enemy. no man is your friend. every man is your teacher. - florence
# scovel shinn
