"""Line up colors in a row.

Use a clustering algorithm to line up colors.

1. Cluster the colors into two clusters.
2. Sort these by the lexicographic order of their exemplars. This is how we will
   break all ties (though others ties are extraordinarily unlikely) to maintain
   determinism.
3. Split each of these into two clusters (if possible).

   A | B

4. For each pair of subclusters, sort them in the order that would minimize the
   distance between them if they were inserted into the position of their parent
   cluster.

   To determine the order of A subclusters, select the shorter of
   ||Ab - B|| < ||Aa - B|| -> Aa Ab | B
   ||Aa - B|| < ||Ab - B|| -> Ab Aa | B

   To determine the order of B subclusters, select the shorter of
   ||Ba - A|| < ||Bb - A|| -> A | Ba Bb
   ||Bb - A|| < ||Ba - A|| -> A | Bb Ba

   Then create a new generation using the subclusters in their determine orders.

   A | B | C | D

5. Repeat the process until no more clusters can be split. When working from more
   than two clusters, keep minimizing the distance from
   A | B | C
   (||A - Ba|| + ||C - Bb||) < (||A - Bb|| + ||C - Ba||) -> A | Ba Bb | C
   (||A - Bb|| + ||C - Ba||) < (||A - Ba|| + ||C - Bb||) -> A | Bb Ba | C

:author: Shay Hill
:created: 2025-05-03
"""

import functools as ft
import random
from collections.abc import Sequence
from operator import attrgetter, itemgetter

from cluster_colors import Cluster, DivisiveSupercluster, Members
from cluster_colors.type_hints import Vectors, VectorsLike

Supercluster = DivisiveSupercluster

colors = [(x % 11 * 20, x // 11 * 20, random.randint(0, 255)) for x in range(0, 100, 5)]

random.shuffle(colors)


def _try_centroid(clusters: Sequence[Cluster], index: int) -> int | None:
    """Return the item at the given index or None if the index is out of range."""
    if index < 0:
        return None
    try:
        return clusters[index].centroid
    except IndexError:
        return None


def _get_proximity(members: Members, ix_a: int | None, ix_b: int | None) -> float:
    if None in (ix_a, ix_b):
        return 0.0
    return float(members.pmatrix[ix_a][ix_b])


def _get_order_error(
    members: Members, beg: int | None, end: int | None, mid_a: int, mid_b: int
) -> float:
    """Return the distance from beg to mid_a plus the distance from mid_b to end.

    Parameter descriptions given an existing palette of color clusters A, B, C. With
    B split into Ba and Bb.

    :param members: A members instance storing a proximity matrix.
    :param beg: The palette index of the centroid of an existing cluster (A) in a
        spectrum or None if there is no cluster before the split cluster.
    :param beg: The palette index of the centroid of an existing cluster (C) in a
        spectrum or None if there is no cluster after the split cluster.
    :param mid_a: The palette index of the centroid of the first subcluster (Ba) of
        the split cluster B
    :param mid_b: The palette index of the centroid of the second subcluster (Bb) of
        the split cluster B
    :return: The distance from A to Ba plus the distance from Bb to C.

    This is used to evaluate one of the two possible orders of the subclusters of B.
    A | Ba Bb | C or A | Bb Ba | C

    If, given the same palette, A or C were split instead of B, the parameters would
    be (None, B, Aa, Ab) or (B, None, Ca, Cb).
    """
    return _get_proximity(members, beg, mid_a) + _get_proximity(members, mid_b, end)


def _atomize_spectrum(spectrum: list[Cluster]) -> None:
    worst = max(spectrum, key=attrgetter("error"))
    if worst.error == 0.0:
        return
    worst_index = spectrum.index(worst)
    infix_order_error = ft.partial(
        _get_order_error,
        worst.members,
        _try_centroid(spectrum, worst_index - 1),
        _try_centroid(spectrum, worst_index + 1),
    )

    new_a, new_b = worst.split()
    mid_a, mid_b = (x.centroid for x in (new_a, new_b))

    if infix_order_error(mid_b, mid_a) < infix_order_error(mid_a, mid_b):
        new_a, new_b = new_b, new_a
    spectrum[worst_index : worst_index + 1] = [new_a, new_b]

    _atomize_spectrum(spectrum)


from typing import Annotated

import numpy as np
from basic_colormath import floats_to_uint8
from numpy import typing as npt


# def atomize_clusters_into_spectrum(clusters: Sequence[Cluster]) -> list[Cluster]:
#     """Atomize a list of clusters into a spectrum.

#     :param clusters: A list of clusters to atomize.
#     :return: A list of clusters that have been atomized into a spectrum.
#     """
#     spectrum = list(clusters)
#     _atomize_spectrum(spectrum)
#     return spectrum


def atomize_into_spectrum(
    colors: VectorsLike,
) -> Annotated[npt.NDArray[np.uint8], (-1, 3)]:
    spectrum = [Cluster.from_vectors(colors)]
    _atomize_spectrum(spectrum)
    spectrum_centroids = [cluster.centroid for cluster in spectrum]
    return floats_to_uint8(spectrum[0].members.vectors[spectrum_centroids])


def by_highest_variance(
    colors: VectorsLike,
) -> Annotated[npt.NDArray[np.uint8], (-1, 3)]:
    """Sort colors by the variance of their RGB values."""
    cluster = Cluster.from_vectors(colors)
    abc = cluster.axis_of_highest_variance

    def rel_dist(x: tuple[float, float, float]) -> float:
        return np.dot(abc, x)

    rgbs = [(r, g, b) for r, g, b in colors]
    rgbs = sorted(rgbs, key=rel_dist)
    return floats_to_uint8(rgbs)


def by_highest_variance_strict(
    colors: VectorsLike, done: list[list[tuple[float, float, float]]] | None = None
) -> Annotated[npt.NDArray[np.uint8], (-1, 3)]:
    if not colors:
        return [(r, g, b) for r, g, b in [x for y in done for x in y]]
    if len(colors) == 1:
        return by_highest_variance_strict([], [*done, colors])
    if done is None:
        done = []

    cluster = Cluster.from_vectors(colors)
    abc = cluster.axis_of_highest_variance

    def rel_dist(x: int) -> float:
        return np.dot(abc, cluster.members.vectors[x])

    ixs = sorted(cluster.ixs, key=rel_dist)
    discard = []
    while True:
        if len(ixs) < 3:
            break
        bad_fits = []
        for i in range(1, len(ixs) - 1):
            to_next = cluster.members.pmatrix[ixs[i - 1]][ixs[i]]
            to_nnext = cluster.members.pmatrix[ixs[i - 1]][ixs[i + 1]]
            delta = to_next - to_nnext
            if delta > 0:
                bad_fits.append((delta, ixs[i]))
        if not bad_fits:
            break
        worst = max(bad_fits, key=itemgetter(0))
        discard.append(worst[1])
        ixs.remove(worst[1])
        print("removing")

    rgbs = [
        (int(r), int(g), int(b))
        for r, g, b in floats_to_uint8(cluster.members.vectors[ixs])
    ]
    colors = [
        (int(r), int(g), int(b))
        for r, g, b in floats_to_uint8(cluster.members.vectors[discard])
    ]
    return colors

    return by_highest_variance_strict(colors, [*done, rgbs])

    print(f"{len(ixs)=}")


def by_highest_variance2(
    colors: VectorsLike,
) -> Annotated[npt.NDArray[np.uint8], (-1, 3)]:
    # if not colors:
    #     return [(r, g, b) for r, g, b in [x for y in done for x in y]]
    # if len(colors) == 1:
    #     return by_highest_variance_strict([], [*done, colors])
    # if done is None:
    #     done = []

    cluster = Cluster.from_vectors(colors)
    abc = cluster.axis_of_highest_variance

    def rel_dist(x: int) -> float:
        return np.dot(abc, cluster.members.vectors[x])

    ixs = sorted(cluster.ixs, key=rel_dist)
    discard = []
    while True:
        if len(ixs) < 3:
            break
        bad_fits = []
        for i in range(1, len(ixs) - 1):
            to_next = cluster.members.pmatrix[ixs[i - 1]][ixs[i]]
            to_nnext = cluster.members.pmatrix[ixs[i - 1]][ixs[i + 1]]
            delta = to_next - to_nnext
            if delta > 0:
                bad_fits.append((delta, ixs[i]))
        if not bad_fits:
            break
        worst = max(bad_fits, key=itemgetter(0))
        discard.append(worst[1])
        ixs.remove(worst[1])
        print("removing")

    ixs_lists = [[x] for x in ixs]
    while len(discard) > 0:
        ixs_x_discard = cluster.members.pmatrix[np.ix_(ixs, discard)]
        s_ix, q_ix = np.unravel_index(np.argmin(ixs_x_discard), ixs_x_discard.shape)
        ixs_lists[s_ix].append(discard[q_ix])
        discard = np.delete(discard, q_ix)
    clusters = [Cluster(members = cluster.members, ixs = x) for x in ixs_lists]
    _atomize_spectrum(clusters)
    ixs = [cluster.centroid for cluster in clusters]

    rgbs = [
        (int(r), int(g), int(b))
        for r, g, b in floats_to_uint8(cluster.members.vectors[ixs])
    ]
    return rgbs
    colors = [
        (int(r), int(g), int(b))
        for r, g, b in floats_to_uint8(cluster.members.vectors[discard])
    ]

    return by_highest_variance_strict(colors, [*done, rgbs])

    print(f"{len(ixs)=}")


#     rgbs = [(r, g, b) for r, g, b in colors]
#     rgbs = sorted(rgbs, key=rel_dist)


aaa = atomize_into_spectrum(colors)
bbb = by_highest_variance(colors)
bbb = by_highest_variance_strict(colors)
ccc = by_highest_variance2(colors)



import svg_ultralight as su
from basic_colormath import rgb_to_hex


def new_bound_rect(color: tuple[int, int, int]) -> su.BoundElement:
    bbox = su.BoundingBox(0, 0, 10, 50)
    rect = su.new_bbox_rect(bbox, fill=rgb_to_hex(color))
    return su.BoundElement(rect, bbox)


rects: list[su.BoundElement] = [new_bound_rect(aaa[0])]
for color in aaa[1:]:
    rects.append(new_bound_rect(color))
    rects[-1].x = rects[-2].x2

rects.append(new_bound_rect(bbb[0]))
rects[-1].y = rects[-2].y2
for color in bbb[1:]:
    rects.append(new_bound_rect(color))
    rects[-1].y = rects[-2].y
    rects[-1].x = rects[-2].x2

rects.append(new_bound_rect(ccc[0]))
rects[-1].y = rects[-2].y2
for color in ccc[1:]:
    rects.append(new_bound_rect(color))
    rects[-1].y = rects[-2].y
    rects[-1].x = rects[-2].x2

root = su.new_svg_root_around_bounds(*rects)
root.extend([x.elem for x in rects])
_ = su.write_svg("test.svg", root)

