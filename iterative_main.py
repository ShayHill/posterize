"""Create a palette with backtracking after each color choice.

This is an edit of fast_main, but it's too much of a departure to create without
tearing fast_main apart.

:author: Shay Hill
:created: 2025-02-06
"""

import enum
import functools
import logging
import pickle
from operator import itemgetter
import copy
import numpy as np
from pathlib import Path
from typing import Annotated, Iterable, Iterator, Sequence, TypeAlias, TypeVar, cast
from contextlib import suppress
import itertools as it
import time

from palette_image.svg_display import write_palette
from palette_image.color_block_ops import sliver_color_blocks

import numpy as np
from basic_colormath import get_delta_e, rgb_to_hsv, hsv_to_rgb, get_delta_e_lab
from cluster_colors import SuperclusterBase, Members
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt

from posterize import paths
from posterize.convolution import shrink_mask
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_supercluster_with_quantized_image

from typing import Any
logging.basicConfig(level=logging.INFO)

# an image-sized array of -1 where transparent and palette indices where opaque
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(n,r,c)"]

_IndexVector: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(n,)"]
_IndexVectorLike: TypeAlias = _IndexVector | Sequence[int]
_LabArray: TypeAlias = Annotated[npt.NDArray[np.floating[Any]], "(n,m,3)"]
_MonoPixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,1)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.floating[Any]], "(n,m,1)"]
_RGBTuple = tuple[int, int, int]
_FPArray = npt.NDArray[np.floating[Any]]

RgbLike = tuple[float, float, float] | Iterable[float]
HsvLike = tuple[float, float, float] | Iterable[float]
HslLike = tuple[float, float, float] | Iterable[float]
LabLike = tuple[float, float, float] | Iterable[float]

# the maximum delta E between two colors in the Lab color space. I'm not sure this
# value is even possible in RGB colorspace, but it should work as a maximum quantify
# functions that are calculated by returning _MAX_DELTA_E - get_anti_value(). Many
# values are easier to quantify as distance from a desireable value, where less
# should be "better".
_MAX_DELTA_E = get_delta_e_lab((0, 127, 127), (100, -128, -128))

WHITE = (255, 255, 255)


class SumSupercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "sum_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"

class Supercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "avg_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"

# _TSuperclusterBase = TypeVar("_TSuperclusterBase", bound=SuperclusterBase)


def _merge_layers(*layers: _IndexMatrix) -> _IndexMatrix:
    """Merge layers into a single layer.

    :param layers: (r, c) arrays with non-negative integers (palette indices) in
        opaque pixels and -1 in transparent
    :return: one (r, c) array with the last non-transparent pixel in each position
    """
    merged = layers[0].copy()
    for layer in layers[1:]:
        merged[np.where(layer != -1)] = layer[np.where(layer != -1)]
    return merged


class TargetImage:
    """A type to store input images, current state, and current cost."""

    def __init__(
        self,
        path: Path,
        bite_size: float | None = None,
    ) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param bite_size: the max error of the cluster removed for each color. If the
            cluster is vibrant, max error will be limited to bite_size. If the
            cluster is not vibrant, average error will be limited to bite_size.
        """
        self._path = path
        self._bite_size = 9 if bite_size is None else bite_size

        self.clusters, self.image = new_supercluster_with_quantized_image(
            Supercluster, path
        )

        # initialize cached properties
        self._layers = np.empty((0,) + self.image.shape[:2], dtype=int)
        self.state = np.ones_like(self.image) * -1
        self.state_cost_matrix = np.ones_like(self.image) * np.inf

    @property
    def layers(self) -> _IndexMatrices:
        return self._layers

    @layers.setter
    def layers(self, value: _IndexMatrices) -> None:
        self._layers = value
        self.state = _merge_layers(*value)
        self.state_cost_matrix = self._get_cost_matrix(self.state)

    @functools.cached_property
    def cache_stem(self) -> str:
        cache_bite_size = f"{self._bite_size:05.2f}".replace(".", "_")
        return f"{self._path.stem}-{cache_bite_size}"

    def get_distribution(self, indices: _IndexVectorLike) -> npt.NDArray[np.intp]:
        """Count the pixels best approximated by each palette index."""
        pmatrix = self.clusters.members.pmatrix
        select_cols = pmatrix[:, indices]
        closest_per_color = np.argmin(select_cols, axis=1)
        image_approx = closest_per_color[self.image]
        return np.bincount(image_approx.flatten(), minlength=len(indices))

    def _get_cost_matrix(self, *layers: _IndexMatrix) -> _ErrorArray:
        """Get the cost-per-pixel between self.image and (state + layers).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        state_with_layers_applied = _merge_layers(self.state, *layers)
        if -1 in state_with_layers_applied:
            msg = "There are still transparent pixels in the state."
            raise ValueError(msg)
        return self.clusters.members.pmatrix[self.image, state_with_layers_applied]

    def get_cost(
        self, *layers: _IndexMatrix, mod: int | None = None
    ) -> tuple[float, float]:
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :param mod: optionally give a modulus number where contrast with layer
            count % mod will be considered instead of contrast with the entire image.
        :return: sum of the cost between image and (state + layer)
        """
        if mod and len(layers) != 1:
            msg = "mod must be None if there are no layers or multiple layers."
            raise ValueError(msg)
        if not layers:
            return (self.state_cost, self.state_cost)
        cost_matrix = self._get_cost_matrix(*layers)
        primary = float(np.sum(cost_matrix))
        fallback = primary

        layer_idx = len(self._layers)
        if mod and layer_idx // mod:
            mask = _merge_layers(*self.layers[:mod])
            cost_matrix[np.where(mask != layer_idx - mod)] = 0
            primary = float(np.sum(cost_matrix))
        return primary, fallback

    @property
    def state_cost(self) -> float:
        """Get the sum of the cost at the current state.

        Use this to guarantee any new element will improve the approximation.
        """
        return float(np.sum(self.state_cost_matrix))

    def new_candidate_layer(self, palette_index: int) -> _IndexMatrix:
        """Create a new candidate state.

        A candidate is the current state with state indices replaced with
        palette_index where palette_index has a lower cost that the index in state at
        that position.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.image.shape, palette_index)
        if self.layers.shape[0] == 0:
            return solid
        solid_cost_matrix = self._get_cost_matrix(solid)
        layer = np.full(self.image.shape, -1)
        layer[np.where(self.state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

    def append_layer(self, layer: _IndexMatrix) -> None:
        """Append a layer to the current state.

        param layer: (m, n) array with a palette index in opaque pixels and -1 in
            transparent
        """
        self.layers = np.append(self.layers, [layer], axis=0)

    def get_colors(self) -> list[int]:
        """Get the most common colors in the image.

        :return: the rough color clusters in the image
        """
        self.clusters.set_max_avg_error(self._bite_size)
        return [x.centroid for x in self.clusters.clusters]

    def get_best_candidate(self, mod: int | None = None) -> _IndexMatrix:
        """Get the best candidate layer.

        :param mod: optionally give a modulus number where contrast with layer
            count % mod will be considered instead of contrast with the entire image.
        :return: the candidate layer with the lowest cost

        """
        candidates = list(map(self.new_candidate_layer, self.get_colors()))
        scored = ((self.get_cost(x, mod=mod), x) for x in candidates)
        return min(scored, key=itemgetter(0))[1]


def pick_nearest_color(
    rgb: tuple[float, float, float], colormap: _FPArray, colors: Iterable[int]
) -> int:
    """Pick the nearest color in colormap to rgb.

    :param rgb: (r, g, b) tuple
    :param colormap: (r, 3) array of colors
    :param colors: indices of colors in colormap to consider
    :return: index of the nearest color in colormap to rgb
    """
    return min(colors, key=lambda x: get_delta_e(rgb, colormap[x]))


def _draw_target(
    target: TargetImage, num_cols: int | None = None, stem: str = ""
) -> None:
    """Infer a name from TargetImage args and write image to file.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    vectors = target.clusters.members.vectors
    stem_parts = (target.cache_stem, len(target.layers), num_cols, stem)
    output_stem = "-".join(_stemize(*stem_parts))
    draw_posterized_image(vectors, target.layers[:num_cols], output_stem)


def _stemize(*args: Path | float | int | str | None) -> Iterator[str]:
    """Convert args to strings and filter out empty strings."""
    if not args:
        return
    if args[0] is None:
        yield from _stemize(*args[1:])
    elif isinstance(args[0], str):
        yield args[0]
        yield from _stemize(*args[1:])
    elif isinstance(args[0], Path):
        yield args[0].stem
        yield from _stemize(*args[1:])
    elif isinstance(args[0], float):
        yield f"{args[0]:05.2f}".replace(".", "_")
        yield from _stemize(*args[1:])
    else:
        assert isinstance(args[0], int)
        yield f"{args[0]:03d}"
        yield from _stemize(*args[1:])


def _new_cache_path(*args: Path | float | int | str | None, suffix: str) -> Path:
    stem = "-".join(_stemize(*args))
    return (paths.CACHE_DIR / stem).with_suffix(suffix)


def posterize(
    image_path: Path,
    bite_size: float,
    ixs: _IndexVectorLike | None = None,
    num_cols: int | None = None,
    mod_cols: int | None = None,
    *,
    ignore_cache: bool = True,
) -> TargetImage:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use. This is used
        with bite_size == 0 to generate an image with exactly the colors in ixs.
    :return: posterized image
    """
    ignore_cache = True
    print(f"{bite_size=}")

    if ixs and num_cols and len(ixs) < num_cols:
        msg = "ixs must be None or have at least as many colors as num_cols."
        raise ValueError(msg)

    target = TargetImage(image_path, bite_size)

    cache_path = _new_cache_path(image_path, bite_size, num_cols, suffix=".npy")
    if cache_path.exists() and not ignore_cache:
        target.layers = np.load(cache_path)
    else:
        if ixs:
            target.clusters = target.clusters.copy(inc_members=ixs)

        while len(target.layers) < (num_cols or 1):
            target.append_layer(target.get_best_candidate(mod=mod_cols))

    np.save(cache_path, target.layers)

    if len(target.layers) < (num_cols or 1):
        return posterize(
            image_path,
            max(bite_size - 1, 0),
            ixs,
            num_cols,
            mod_cols,
            ignore_cache=ignore_cache,
        )

    return target


def _compress_to_n_colors(
    target: TargetImage, net_cols: int, gross_cols: int | None = None
):
    state_at_n = _merge_layers(*target.layers[:gross_cols])


class Mood(str, enum.Enum):
    """Mood of the image."""

    VIBRANT = "vibrant"
    MUTED = "muted"
    DEEP = "deep"
    MARSS = "marss"
    NEUTRAL = "neutral"

    COLORFUL = "colorful"
    CONTRAST = "contrast"
    THRIFTY = "thrifty"
    FAITHFUL = "faithful"


def _purify_color(rgb: RgbLike) -> tuple[float, float, float]:
    """Purify a color by converting it to HSV and back to RGB.

    :param rgb: (r, g, b) color
    :return: purified (r, g, b) color
    """
    r, g, b = rgb
    hsv = rgb_to_hsv((r, g, b))
    hsv = (hsv[0], 100, 100)
    return hsv_to_rgb(hsv)


def _desaturate_color(rgb: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Desaturate a color by multiplying by constants."""
    r, g, b = rgb
    gray = _get_brightness(rgb)
    return gray, gray, gray


def _get_dullness(rgb: npt.NDArray[np.float64]) -> float:
    """Get the vibrance of a color.

    :param rgb: (r, g, b) color
    :return: vibrance of the color
    """
    r, g, b = rgb
    return get_delta_e((r, g, b), _purify_color(rgb))


def _get_brightness(rgb: RgbLike) -> float:
    """Get the brightness of a color.

    :param rgb: (r, g, b) color
    :return: brightness of the color
    """
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _match_brightness(rgb: RgbLike, reference: RgbLike) -> tuple[float, float, float]:
    """Match the brightness of a color to a reference brightness.

    :param rgb: (r, g, b) color
    :param reference_brightness: brightness to match
    :return: color with the same brightness as reference_brightness
    """
    reference_brightness = _get_brightness(reference)
    pure_color = _purify_color(rgb)

    now_brightness = _get_brightness(pure_color)

    if now_brightness >= reference_brightness:
        scale = reference_brightness / now_brightness
        r, g, b = (x * scale for x in pure_color)
        return r, g, b

    scale = (reference_brightness - 255) / (now_brightness - 255)
    r, g, b = (x * scale + 255 * (1 - scale) for x in pure_color)
    return r, g, b


def _qtfy_deep(rgb: RgbLike) -> float:
    """Quantify the depth of a color."""
    deep = _match_brightness(rgb, (0, 0, 255))
    return _MAX_DELTA_E - get_delta_e(rgb, deep)


def _qtfy_mute(rgb: RgbLike) -> float:
    mute = _match_brightness(rgb, (255, 255, 0))
    return _MAX_DELTA_E - get_delta_e(rgb, mute)


def _qtfy_marss(rgb: RgbLike) -> float:
    marss = _match_brightness(rgb, (0, 140, 140))
    return _MAX_DELTA_E - get_delta_e(rgb, marss)


def _qtfy_vibrant(rgb: RgbLike) -> float:
    vibrant = _purify_color(rgb)
    return _MAX_DELTA_E - get_delta_e(rgb, vibrant)


def _qtfy_gray(rgb: RgbLike) -> float:
    gray = _get_brightness(rgb)
    return _MAX_DELTA_E - get_delta_e(rgb, (gray, gray, gray))


def _qtfy_neutral(rgb: RgbLike) -> float:
    deep = _qtfy_deep(rgb)
    mute = _qtfy_mute(rgb)
    gray = _qtfy_gray(rgb)
    marss = _qtfy_marss(rgb)
    return max(deep, mute, gray, marss)


def _get_saturation(rgb: npt.NDArray[np.float64]) -> float:
    """Get the saturation of a color.

    :param rgb: (r, g, b) color
    :return: saturation of the color
    """
    r, g, b = rgb
    return get_delta_e((r, g, b), _desaturate_color(rgb))


def _get_radial_distance(rgb_a: RgbLike, rgb_b: RgbLike) -> float:
    """Get the radial distance between two hues.

    :param hue_a: hue in degrees
    :param hue_b: hue in degrees
    :return: radial distance between hue_a and hue_b
    """
    hue_a = rgb_to_hsv(rgb_a)[0]
    hue_b = rgb_to_hsv(rgb_b)[0]
    return min(abs(hue_a - hue_b), 360 - abs(hue_a - hue_b))


def _qtfy_colorful(rgb: RgbLike) -> float:
    mute = _qtfy_mute(rgb)
    vibrant = _qtfy_vibrant(rgb)
    marss = _qtfy_marss(rgb)
    return max(mute, vibrant, marss)


def _get_palette_color_cost(
    members: Members,
    palette: list[int | None],
    mood: Mood,
    idxs: Iterable[int],
):
    idxs = sorted(idxs)
    choices = [(r, g, b) for r, g, b in (members.vectors[x] for x in idxs)]
    if mood == Mood.VIBRANT:
        choice = max(choices, key=_qtfy_vibrant)
        return idxs[choices.index(choice)]
    if mood == Mood.DEEP:
        choice = max(choices, key=_qtfy_deep)
        return idxs[choices.index(choice)]
    if mood == Mood.MARSS:
        choice = max(choices, key=_qtfy_marss)
        return idxs[choices.index(choice)]
    if mood == Mood.MUTED:
        choice = max(choices, key=_qtfy_mute)
        return idxs[choices.index(choice)]
    if mood == Mood.NEUTRAL:
        choice = max(choices, key=_qtfy_neutral)
        return idxs[choices.index(choice)]

    if mood == Mood.CONTRAST:
        choice = min(choices, key=_qtfy_vibrant)
        return idxs[choices.index(choice)]

    if mood == Mood.COLORFUL:
        choice = min(choices, key=_qtfy_vibrant)
        return idxs[choices.index(choice)]

    else:
        msg = f"mood {mood} not implemented"
        raise NotImplementedError(msg)


def _separate_colors(layers: list[npt.NDArray[np.int32]]) -> list[set[int]]:
    """Group colors by layer in which they are most common.

    :param layers:

    Nothing will blow up if some of the color sets end up empty. Downstream of this,
    all colors are searched with no layer color is found.
    """
    all_cols: set[int] = set().union(*(np.unique(x) for x in layers))
    cols_per_layer: list[set[int]] = [set() for _ in range(len(layers))]
    for color in all_cols:
        counts = [np.sum(layer == color) for layer in layers]
        cols_per_layer[counts.index(max(counts))].add(color)
    return cols_per_layer


def _elect_palette(*palettes: list[int]) -> list[int | None]:
    """Assign a value to each palette index if it represents a majority.

    :param palettes: lists of palette indices
    :return: list of palette indices
    """
    elected: list[int | None] = []
    majority = len(palettes) // 2 + 1
    for channel in zip(*palettes):
        values = {x for x in channel if x is not None}
        winner = max(values, key=lambda x: channel.count(x))
        if channel.count(winner) >= majority:
            elected.append(winner)
        else:
            elected.append(None)
    return elected


def _select_vibrant(
    vectors: npt.NDArray[np.float64], cols: set[int], min_required: int
) -> set[int]:
    """Select the most vibrant colors.

    :return: a set of palette indices that are considered vibrant

    These colors cannot be removed from `self._clusters.ixs` unless they are
    within `bite_size` of the cluster centroid. This prevents vibrant outliers
    from disappearing into less-vibrant clusters.

    Will take the 10% most vibrant colors in the image, but will exclude colors
    with vibrance < 64. This is to prevent muddy colors from being considered
    vibrant just because an image is mostly grayscale.
    """
    scored = [(_qtfy_vibrant(vectors[x]), x) for x in cols]
    median = np.median([s for s, _ in scored])
    vibrant = {x for s, x in scored if s >= median}
    if len(vibrant) < min_required:
        return set([x for _, x in sorted(scored)][-min_required:])
    return vibrant


def _qtfy_contrast(vectors: _FPArray, col_idxs: tuple[int, ...]) -> float:
    """Quantify the contrast between colors in a palette.

    :param vectors: (n, 3) array of colors
    :param col_idxs: indices of colors in the palette
    :return: contrast between colors in the palette
    """
    col_vectors = [vectors[x] for x in col_idxs]
    return sum(get_delta_e(x, y) for x, y in it.combinations(col_vectors, 2))


def _qtfy_hue_contrast(vectors: _FPArray, col_idxs: tuple[int, ...]) -> float:
    """Quantify the hue contrast between colors in a palette.

    :param vectors: (n, 3) array of colors
    :param col_idxs: indices of colors in the palette
    :return: contrast between colors in the palette
    """
    col_vectors = [vectors[x] for x in col_idxs]
    return sum(_get_radial_distance(x, y) for x, y in it.combinations(col_vectors, 2))


def _iter_candidates(
    palette: list[int | None], colors: set[int], num_cols: int
) -> Iterator[tuple[int, ...]]:
    """Yield candidate palettes given an incomplete palette and fill colors."""
    known = tuple(filter(None, palette))
    for candidate in it.product(colors - set(known), repeat=num_cols - len(known)):
        if len(set(candidate)) == len(candidate):
            yield known + candidate


def _get_subset_weights(members: Members, ixs: _IndexVectorLike) -> list[float]:
    """Get the cumulative weight of vectors at index and their nearest neighbors."""
    ixs = sorted(ixs)
    subset_cols = members.pmatrix[:, ixs]
    nearest_per_row = np.argmin(subset_cols, axis=1)
    nearest_rows_per_idx = (np.where(nearest_per_row == i) for i in range(len(ixs)))
    return [np.sum(members.weights[x]) for x in nearest_rows_per_idx]


def re_weigh(members: Members, ixs: _IndexVectorLike) -> Members:
    ixs = sorted(ixs)
    subset_vectors = members.vectors[ixs]
    subset_weights = _get_subset_weights(members, ixs)
    subset_pmatrix = members.pmatrix[np.ix_(ixs, ixs)]
    return Members(subset_vectors, weights=subset_weights, pmatrix=subset_pmatrix)


def posterize_to_n_colors(
    image_path: Path,
    ixs: _IndexVectorLike,
    bite_size: float,
    num_cols: int,
    mood: Mood = Mood.FAITHFUL,
    seen: set[tuple[int, ...]] | None = None,
    pick_: list[int | None] | None = None,
    min_dist: float = 16,
) -> list[int] | None:

    print(f"{image_path.stem} {mood} {min_dist}")
    bite_size = 24
    while bite_size >= 0:
        target = TargetImage(image_path, bite_size)
        vectors = target.clusters.members.vectors
        new_ixs = target.clusters.ixs
        new_ixs = [x for x in new_ixs if min(vectors[x]) < 90]
        new_ixs = [x for x in new_ixs if max(vectors[x]) > 90]
        new_ixs = [x for x in new_ixs if rgb_to_hsv(vectors[x])[1] > 40]

        vs = [tuple(map(int, vectors[x])) for x in new_ixs]
        new_ixs_array = np.array(new_ixs, dtype=np.int32)
        target.clusters = target.clusters.copy(inc_members=new_ixs_array)
        target = posterize(image_path, 9, ixs, 12, 6, ignore_cache=False)
        _draw_target(target, 6, "input_06")
        _draw_target(target, 12, "input_12")
        # _draw_target(target, 18, "input_12")
        break

    target = posterize(image_path, 9, new_ixs, 12, 6, ignore_cache=False)

    kept = np.unique(_merge_layers(*target.layers[:12]))
    vss_2 = [tuple(map(int, vectors[x])) for x in kept]
    assert not set(vss_2) - set(vs)

    members = re_weigh(target.clusters.members, kept)
    clusters = SumSupercluster(members)

    clusters.set_n(6)
    palette = [x.centroid for x in clusters.clusters]
    # palette = [kept[x] for x in palette]
    # target = posterize(image_path, 0, palette, ignore_cache=True)
    # _draw_target(target, 6, "input_selected")

    vectors = members.vectors

    dist = target.get_distribution(palette)
    vss2 = [tuple(map(int, vectors[x])) for x in palette]
    color_blocks = sliver_color_blocks(vectors[palette], list(map(float, dist)))
    output_name = paths.WORKING / f"{image_path.stem}-{mood}.svg"

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
        seen: set[tuple[int, ...]] = set()
        pos = functools.partial(
            posterize_to_n_colors,
            image_path,
            bite_size=9,
            ixs=(),
            num_cols=6,
            seen=seen,
        )

        cached: dict[Mood, list[int]] = {}
        for mood in Mood:
            mood = Mood.CONTRAST
            print(f"==============={pic} -- {mood}")
            if mood == Mood.CONTRAST:
                moods = (Mood.VIBRANT, Mood.MUTED, Mood.MARSS, Mood.DEEP, Mood.NEUTRAL)
                voters = filter(None, (cached.get(x) for x in moods))
                pick = _elect_palette(*voters)
            if mood == Mood.COLORFUL:
                moods = (Mood.VIBRANT, Mood.MUTED, Mood.MARSS)
                voters = filter(None, (cached.get(x) for x in moods))
                pick = _elect_palette(*voters)
            else:
                pick = None
            with suppress(NotImplementedError):
                palette = pos(mood=mood, pick_=pick)
                if palette is not None:
                    cached[mood] = palette
            break

    print("done")

# industry, perseverance, and frugality # make fortune yield - benjamin franklin
# strength, well being, and health
# no man is your enemy. no man is your friend. every man is your teacher. - florence
# scovel shinn
