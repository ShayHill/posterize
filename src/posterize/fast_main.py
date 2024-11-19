import enum
import functools
import logging
import pickle
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Iterable, Iterator, Sequence, TypeAlias, TypeVar, cast
from contextlib import suppress
import itertools as it

import numpy as np
from basic_colormath import get_delta_e, rgb_to_hsv, hsv_to_rgb, get_delta_e_lab
from cluster_colors import SuperclusterBase, Members
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt

from posterize import paths
from posterize.convolution import shrink_mask
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_supercluster_with_quantized_image

logging.basicConfig(level=logging.INFO)

# an image-sized array of -1 where transparent and palette indices where opaque
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int32], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.int32], "(n,r,c)"]

_IndexVector: TypeAlias = Annotated[npt.NDArray[np.int32], "(n,)"]
_IndexVectorLike: TypeAlias = _IndexVector | Sequence[int]
_LabArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,3)"]
_MonoPixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,1)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,1)"]
_RGBTuple = tuple[int, int, int]
_FPArray = npt.NDArray[np.float64]

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
        *,
        ignore_cache: bool = False,
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

    def get_cost(self, *layers: _IndexMatrix) -> float:
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :return: sum of the cost between image and (state + layer)
        """
        if layers:
            return float(np.sum(self._get_cost_matrix(*layers)))
        return self.state_cost

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

    def _split_to_exclude_vibrant(self, palette_index: int):
        """Exclude vibrant colors from a cluster.

        Split clusters until the cluster containing palette_index does not contain
        any vibrant outliers. This protects vibrant colors from disappearing into
        less-vibrant clusters.
        """
        if palette_index in self.vibrant_colors:
            return

        pmatrix = self.clusters.members.pmatrix
        while True:
            cluster_ix = self.clusters.find_member(palette_index)
            cluster = self.clusters.clusters[cluster_ix]
            vibrant_in_cluster = list(self.vibrant_colors & set(cluster.ixs))
            if not vibrant_in_cluster:
                break
            vibrant_in_cluster_proximities = pmatrix[palette_index, vibrant_in_cluster]
            if max(vibrant_in_cluster_proximities) < self._bite_size:
                break
            self.clusters.split()

    def append_layer(self, layer: _IndexMatrix) -> None:
        """Append a layer to the current state.

        :param layer: (m, n) array with a palette index in opaque pixels and -1 in
            transparent
        """
        self.layers = np.append(self.layers, [layer], axis=0)

        layer_color = max(np.unique(layer))
        assert layer_color != -1

        self.clusters.set_max_avg_error(self._bite_size)
        self._split_to_exclude_vibrant(layer_color)
        self.clusters = self.clusters.copy(exc_member_clusters=[layer_color])
        print(f"####### {len(self.clusters.ixs)} members remain")

    @functools.cached_property
    def vibrant_colors(self) -> set[int]:
        """Get the vibrant colors in the image.

        :return: a set of palette indices that are considered vibrant

        These colors cannot be removed from `self._clusters.ixs` unless they are
        within `bite_size` of the cluster centroid. This prevents vibrant outliers
        from disappearing into less-vibrant clusters.

        Will take the 10% most vibrant colors in the image, but will exclude colors
        with vibrance < 64. This is to prevent muddy colors from being considered
        vibrant just because an image is mostly grayscale.
        """
        vectors = self.clusters.members.vectors
        v_max = cast(npt.NDArray[np.float64], np.max(vectors, axis=1))
        v_min = cast(npt.NDArray[np.float64], np.min(vectors, axis=1))
        v_vib = v_max - v_min
        threshold = max(float(np.percentile(v_vib, 90)), 64)
        vibrant = np.where(v_vib >= threshold)[0]
        print(f"vibrant colors: {len(vibrant)}")
        return set(map(int, vibrant))

    def get_colors(self) -> list[int]:
        """Get the most common colors in the image.

        :return: the rough color clusters in the image
        """
        self.clusters.set_max_avg_error(self._bite_size)
        print(f"generated {len(self.clusters.clusters)} colors")
        return [x.centroid for x in self.clusters.clusters]

    def get_best_candidate(self) -> _IndexMatrix:
        """Get the best candidate layer.

        :return: the candidate layer with the lowest cost
        """
        candidates = map(self.new_candidate_layer, self.get_colors())
        scored = ((self.get_cost(x), x) for x in candidates)
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
    stem_parts = (target.cache_stem, len(target.clusters.ixs), num_cols, stem)
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
    bite_size: float | None = None,
    ixs: _IndexVectorLike | None = None,
    num_cols: int | None = None,
    mood: str | None = None,
    *,
    ignore_cache: bool = True,
) -> TargetImage:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use
    :return: posterized image
    """
    target = TargetImage(image_path, bite_size, ignore_cache=ignore_cache)

    cache_path = _new_cache_path(image_path, bite_size, num_cols, suffix=".npy")
    if cache_path.exists() and not ignore_cache:
        target.layers = np.load(cache_path)
        return target

    if ixs:
        target.clusters = target.clusters.copy(inc_members=ixs)
    while len(target.clusters.ixs) > 0:
        target.append_layer(target.get_best_candidate())
    print(f"appended {len(target.layers)} layers")

    _draw_target(target, num_cols, mood or "")

    np.save(cache_path, target.layers)

    return target


def _compress_to_n_colors(
    target: TargetImage, net_cols: int, gross_cols: int | None = None
):
    state_at_n = _merge_layers(*target.layers[:gross_cols])


class Mood(str, enum.Enum):
    """Mood of the image."""

    COLORFUL = "colorful"
    CONTRAST = "contrast"
    VIBRANT = "vibrant"
    MUTED = "muted"
    DEEP = "deep"
    MARSS = "marss"
    NEUTRAL = "neutral"
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


def _get_radial_distance(hue_a: float, hue_b: float) -> float:
    """Get the radial distance between two hues.

    :param hue_a: hue in degrees
    :param hue_b: hue in degrees
    :return: radial distance between hue_a and hue_b
    """
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
    # if mood == Mood.COLORFUL:
    #     known = [members.vectors[x] for x in palette]
    #     known = [x for x in known if _get_saturation(x) > 10]
    #     color = members.vectors[idx]
    #     if not known:
    #         return -_qtfy_colorful(color)
    #     if _get_saturation(color) < 10:
    #         return 100
    #     color_hue = rgb_to_hsv(color)[0]
    #     mrd = min(_get_radial_distance(color_hue, rgb_to_hsv(x)[0]) for x in known)
    #     return -mrd * _qtfy_colorful(color)

    if mood == Mood.COLORFUL:
        choice = max(choices, key=_qtfy_colorful)
        return idxs[choices.index(choice)]
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


def _elect_palette(*palettes: list[int | None]) -> list[int | None]:
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


def posterize_to_n_colors(
    image_path: Path,
    ixs: _IndexVectorLike,
    bite_size: float,
    num_cols: int,
    mood: Mood = Mood.FAITHFUL,
    seen: set[tuple[int, ...]] | None = None,
    pick_: list[int | None] | None = None,
) -> list[int]:

    seen = set() if seen is None else seen
    print(seen)

    ixs_ = () if ixs is None else tuple(ixs)
    target = posterize(image_path, bite_size, ixs_, num_cols, None, ignore_cache=False)
    state_copy = target.state.copy()
    vectors = target.clusters.members.vectors

    state_at_n = _merge_layers(*target.layers[:num_cols])
    net_colors = [x for x in np.unique(state_at_n) if x != -1]
    masks = [np.where(state_at_n == x, 1, 0) for x in net_colors]

    if mood == Mood.FAITHFUL:
        _ = posterize(image_path, 0, net_colors, mood=mood)
        return net_colors

    masks = [shrink_mask(m, 1) for m in masks]
    colors_per_mask = _separate_colors([state_copy[np.where(m == 1)] for m in masks])

    pick: list[int | None] = pick_ or [None] * num_cols
    for _ in range(num_cols):

        get_cost = functools.partial(
            _get_palette_color_cost,
            target.clusters.members,
            pick,
            mood,
        )

        pick_vecs = [vectors[p] for p in pick if p is not None]

        def get_palette_prox(col: int) -> float:
            col_vec = vectors[col]
            return min((get_delta_e(col_vec, p) for p in pick_vecs), default=np.inf)

        def sufficient_contrast(col: int) -> bool:
            return get_palette_prox(col) > 16

        open_idxs = [j for j, p in enumerate(pick) if p is None]
        cols = set(it.chain(*(colors_per_mask[i] for i in open_idxs)))
        cols = set(filter(sufficient_contrast, cols))
        if not cols:
            cols = set(filter(sufficient_contrast, it.chain(*colors_per_mask)))
        # TODO: raise and catch a different error
        if not cols:
            raise NotImplementedError

        best_col = get_cost(cols)
        idx = next(j for j, p in enumerate(colors_per_mask) if best_col in p)
        if pick[idx] is None:
            pick[idx] = best_col
        else:
            pick.append(best_col)

    # TODO: raise and catch a different error
    palette = list(filter(None, pick))
    if len(palette) < num_cols:
        raise NotImplementedError

    if tuple(palette) in seen:
        raise NotImplementedError
    seen.add(tuple(palette))

    _ = posterize(image_path, 0, tuple(palette), mood=mood)
    return palette


if __name__ == "__main__":
    pics = [
        "cafe_at_arles.jpg",
        "manet.jpg",
        "adidas.jpg",
        "broadway.jpg",
        "starry_night.jpg",
        "eyes.jpg",
        "you_the_living.jpg",
        "pencils.jpg",
        "dutch.jpg",
        "blue.jpg",
        "Johannes Vermeer - The Milkmaid.jpg",
        "Johannes Vermeer - Girl with a Pearl Earring.jpg"
    ]
    for pic in pics:
        image_path = paths.PROJECT / f"tests/resources/{pic}"
        if not image_path.exists():
            print(f"skipping {image_path}")
            continue
        seen: set[tuple[int, ...]] = set()
        pos = functools.partial(
            posterize_to_n_colors,
            image_path,
            bite_size=9,
            ixs=(),
            num_cols=6,
            seen=seen,
        )

        for mood in Mood:
            with suppress(NotImplementedError):
                _ = pos(mood=mood)

    print("done")
