"""Create a palette with backtracking after each color choice.

Starting from a quantized image, select colors, one by one, that best approximate
that image. The first color will be a solid layer. Additional layers will have a
single color index where that color would improve the image, and -1 where it would
not.

After layers are added, check the original layers and update their colors if it
improves the approximation. To understand how this would improve an approximation,
imagine a Japanese flag. The algorithm would begin by selecting a color that
minimizes the error across the entire image. This would be some shade of pink. The
second layer would only cover part of the image and would be white or red. Let's say
it's red. Once red is added, the underlying pink layer is no longer responsible for
approximating the red circle in the middle of the flag. So the pink layer can be
updated to white. This saves a third white layer which, combined with the red layer,
would completely cover the pink layer anyway.

:author: Shay Hill
:created: 2025-02-06
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, cast

import diskcache
import numpy as np
from numpy import typing as npt
from PIL import Image

from posterize.color_attributes import get_vibrance
from posterize.layers import apply_mask, merge_layers
from posterize.posterization import Posterization
from posterize.quantization import TargetImage, new_target_image, new_target_image_mono

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


cache = diskcache.Cache(".cache_posterize")

_IntA: TypeAlias = npt.NDArray[np.intp]
_FltA: TypeAlias = npt.NDArray[np.float64]

# Default weight for sum savings vs. average savings. Average savings is, by default,
# weighted highly. These values are used when selecting the best candidate for the
# next layer color. A higher average savings weight means colors that improve the
# approximation a lot in a small area are chosen over colors that improve the
# approximation a tiny amount over a large area. _DEFAULT_SAVINGS_WEIGHT is the
# weight given to sum savings. (1 - _DEFAULT_SAVINGS_WEIGHT) is the weight given to
# average savings.
_DEFAULT_SAVINGS_WEIGHT = 0.25


# A higher number (1.0 is maximum) means colors that are more vibrant are more likely
# to be selected as layer colors. The default is 0.0, which will be good for most
# images, but the parameter is available if you have an overall drab image with a few
# bright highlights and want to pay less attention to the background.
_DEFAULT_VIBRANT_WEIGHT = 0.0


class ColorsExhaustedError(Exception):
    """Exception raised when a new layer is requested, but no colors are available."""

    def __init__(self, message: str = "No available colors.") -> None:
        """Initialize the ColorsExhaustedError exception."""
        self.message = message
        super().__init__(self.message)


def _set_max_val_to_one(floats: Iterable[float]) -> _FltA:
    """Scale an array so the maximum value is 1.

    :param array: the array to scale
    :return: the scaled array

    This is used to scale layer savings and layer weights (all positive numbers) to a
    similar range before taking a weighted average. I did not use min-max scaling
    because I would prefer layers with similar savings or weights to be treated as
    roughly equivalent. I don't want to treat one as 1 (full weight) and the other as
    0 (no weight) when they are very nearly equal.

    The scaling here allows for values to cluster around 1.0 if the input values are
    all high.
    """
    array = np.array(list(floats))
    max_val = np.max(array)
    if max_val == 0:
        return array
    return array / max_val


class ImageApproximation:
    """State for an image approximation.

    :param target_image: the quantized image to approximate
    :param colors: the subset of color indices (TargetImage.clusters.ixs) available
        for use in layers.
    :param layers: (n, c) array of n layers, each containing a value (color index) in
        colors and -1 for transparent. The first layer will be a solid color and
        contain no -1 values. This is a parameter for algorithms that might want to
        restart an ImageApproximation from a previous state, but you can for the most
        part ignore it.
    :param savings_weight: optional kwarg only param - weight for sum savings vs
        average savings when selecting a layer color.
    :param vibrant_weight: optional kwarg only param - [0.0, 1.0] increase tendency
        to select vibrant colors.
    """

    def __init__(
        self,
        target_image: TargetImage,
        colors: Iterable[int] | None = None,
        layers: _IntA | None = None,
        *,
        savings_weight: float | None = None,
        vibrant_weight: float | None = None,
    ) -> None:
        """Initialize the ImageApproximation state."""
        self.target = target_image
        if colors is None:
            self.colors = tuple(range(len(target_image.palette)))
        else:
            self.colors = tuple(colors)
        if layers is None:
            self.layers = np.empty((0, len(target_image.palette)), dtype=np.intp)
        else:
            self.layers = layers
        self.savings_weight = savings_weight or _DEFAULT_SAVINGS_WEIGHT
        self.vibrant_weight = vibrant_weight or _DEFAULT_VIBRANT_WEIGHT

        palette = cast("Iterable[npt.NDArray[np.uint8]]", self.target.palette)
        vibrancies = np.array([get_vibrance(c) for c in palette])
        self.target.weights = (
            self.target.weights * (1 - self.vibrant_weight)
            + vibrancies * self.vibrant_weight
        )

    @property
    def layer_colors(self) -> list[int]:
        """Get the non-transparent color in each layer."""
        return [int(np.max(x)) for x in self.layers]

    def get_layer_masks(self) -> _IntA:
        """Get the masks for each layer.

        :return: (n, r, c) array of masks for each layer
        """
        image = merge_layers(*self.layers)
        return np.array([np.where(image == x, 1, 0) for x in self.layer_colors])

    def get_available_colors(self) -> list[int]:
        """Get available colors in the image."""
        if len(self.layers) == 0:
            return list(self.colors)
        used_colors = set(self.layer_colors)
        return [x for x in self.colors if x not in used_colors]

    def _add_one_layer(self, mask: _IntA | None = None) -> None:
        """Add one layer to the state."""
        new_layer = self.get_best_candidate_layer(mask=mask)
        self.layers = np.concatenate([self.layers, [new_layer]])

    def two_pass_fill_layers(self, num_layers: int) -> None:
        """Fill layers to create masks then fill masks to create layers.

        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers

        The first pass is only to create masks. The colors selected in the first pass
        will try to approximate a larger area than the mask will cover. This creates
        the Japanese flag effect described in the module docstring.

        The second pass selects colors that only try to approximate the mask area.
        """
        if len(self.layers) >= num_layers:
            return
        try:
            self._add_one_layer()
        except ColorsExhaustedError:
            return
        if len(self.layers) >= 2:
            layer_masks = self.get_layer_masks()
            self.layers.resize((0, self.layers.shape[1]), refcheck=False)
            for mask in layer_masks:
                self._add_one_layer(mask=mask)
        self.two_pass_fill_layers(num_layers)

    # ===============================================================================
    #   Define and select new candidate layers
    # ===============================================================================

    def _new_candidate_layer(
        self,
        palette_index: int,
        state_cost_matrix: npt.NDArray[np.floating[Any]] | None = None,
    ) -> _IntA:
        """Create a new candidate layer.

        :param palette_index: the index of the color to use in the new layer
        :param state_cost_matrix: pre-computed state cost matrix to avoid recalculation
        :return: a new candidate layer with the same shape as the quantized image.
            Pixels where palette_index would improve the approximation are set to
            palette_index. Pixels where palette_index would not improve the
            approximation are set to -1.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.ones(self.target.pmatrix.shape[0], dtype=int) * palette_index
        if len(self.layers) == 0:
            return solid

        solid_cost_matrix = self.target.get_cost_matrix(solid)
        if state_cost_matrix is None:
            state_cost_matrix = self.target.get_cost_matrix(*self.layers)
        return np.where(state_cost_matrix > solid_cost_matrix, palette_index, -1)

    def _sum_masked_weight(self, layer: _IntA, mask: None | _IntA) -> np.float64:
        """Mask a layer then sum the weights.

        This is a subroutine for get_best_candidate_layer.
        """
        masked = apply_mask(layer, mask)
        weights = self.target.weights[np.where(masked != -1)]
        return np.sum(weights)

    def get_best_candidate_layer(self, mask: _IntA | None = None) -> _IntA:
        """Get the best candidate layer to add to layers.

        :param state_layers: the current state or a presumed state
        :return: the candidate layer with the lowest cost
        """
        state = merge_layers(*self.layers)
        state_cost = self.target.get_cost(state, mask=mask)
        available_colors = self.get_available_colors()

        if not available_colors:
            raise ColorsExhaustedError

        state_cost_matrix = None
        if len(self.layers) > 0:
            state_cost_matrix = self.target.get_cost_matrix(*self.layers)

        candidates = [
            self._new_candidate_layer(x, state_cost_matrix) for x in available_colors
        ]
        scores = [self.target.get_cost(state, x, mask=mask) for x in candidates]

        if state_cost == np.inf:
            state_cost = max(scores)

        layer_savings = _set_max_val_to_one([state_cost - x for x in scores])
        layer_weights = _set_max_val_to_one(
            [self._sum_masked_weight(x, mask) for x in candidates]
        )
        layer_averages = [
            float(s / p) if p > 0 else 0.0
            for s, p in zip(layer_savings, layer_weights, strict=True)
        ]

        more_is_better = [
            s * self.savings_weight + a * (1 - self.savings_weight)
            for s, a in zip(layer_savings, layer_averages, strict=True)
        ]

        best_idx = np.argmax(np.array(more_is_better, dtype=np.float64))
        return candidates[best_idx]


@cache.memoize()
def posterize(
    image_path: str | os.PathLike[str],
    num_cols: int,
    *,
    savings_weight: None | float = None,
    vibrant_weight: None | float = None,
    max_dim: None | int = None,
) -> Posterization:
    """Posterize an image.

    :param image_path: path to the image
    :param num_cols: the number of colors in the posterization image. If not enough
        colors are available, will silently return fewer layers / colors.
    :param savings_weight: weight for the savings metric vs average savings
    :param vibrant_weight: weight for the vibrance metric vs savings metric
    :param ignore_quantized_image_cache: if True, ignore any cached quantized image
        results
    :param resolution: if not None, resize the image to have a maximum dimension
        of this value before processing. The resized image is not written to disk.
    :return: posterized image result
    """
    target = new_target_image(image_path, max_dim)
    state = ImageApproximation(
        target, savings_weight=savings_weight, vibrant_weight=vibrant_weight
    )
    state.two_pass_fill_layers(num_cols)
    return Posterization(target.indices, target.palette, state.layers)


@cache.memoize()
def posterize_mono(
    pixels: Annotated[npt.NDArray[np.uint8], "(r, c)"],
    num_cols: int,
    *,
    savings_weight: None | float = None,
    vibrant_weight: None | float = None,
) -> Posterization:
    """Posterize a monochrome (r, c) uint8 array.

    :param pixels: (r, c) array of uint8 values (e.g. grayscale)
    :param num_cols: number of colors in the posterization
    :param savings_weight: weight for the savings metric vs average savings
    :param vibrant_weight: weight for the vibrance metric vs savings metric
    :return: posterized result
    """
    target = new_target_image_mono(pixels)
    state = ImageApproximation(
        target, savings_weight=savings_weight, vibrant_weight=vibrant_weight
    )
    state.two_pass_fill_layers(num_cols)
    return Posterization(target.indices, target.palette, state.layers)


def this_is_all_i_need_to_do():
    posterized = posterize("chaucer.png", 9)
    _ = posterized.write_svg("chaucer_posterized.svg")

    image = Image.open("chaucer.png")
    mono = np.array(image)[:, :, 0]
    posterized = posterize_mono(mono, 9)
    _ = posterized.write_svg("chaucer_posterized_mono.svg")


if __name__ == "__main__":
    this_is_all_i_need_to_do()
