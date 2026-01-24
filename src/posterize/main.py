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

from typing import TYPE_CHECKING, Annotated, TypeAlias, cast

import numpy as np
from numpy import typing as npt

from posterize.color_attributes import get_vibrance
from posterize.layers import apply_mask, merge_layers
from posterize.quantization import TargetImage, new_target_image

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


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
        self.cached_states: dict[tuple[int, ...], float] = {}
        self.savings_weight = savings_weight or _DEFAULT_SAVINGS_WEIGHT
        self.vibrant_weight = vibrant_weight or _DEFAULT_VIBRANT_WEIGHT

        palette = cast("Iterable[npt.NDArray[np.uint8]]", self.target.palette)
        vibrancies = np.array(list(map(get_vibrance, palette)))
        self.target.weights *= 1 - self.vibrant_weight
        vibrancies *= self.vibrant_weight
        self.target.weights += vibrancies

    @property
    def layer_colors(self) -> list[int]:
        """Get the non-transparent color in each layer."""
        return [int(np.max(x)) for x in self.layers]

    def get_layer_masks(self) -> _IntA:
        """Get the masks for each layer.

        :return: (n, r, c) array of masks for each layer
        """
        layers = cast("Iterable[npt.NDArray[np.intp]]", self.layers)
        image = merge_layers(*layers)
        return np.array([np.where(image == x, 1, 0) for x in self.layer_colors])

    def get_available_colors(self) -> list[int]:
        """Get available colors in the image."""
        if len(self.layers) == 0:
            return list(self.colors)
        layer_colors = self.layer_colors
        return [x for x in self.colors if x not in layer_colors]

    def _add_one_layer(self, mask: _IntA | None = None) -> None:
        """Add one layer to the state."""
        new_layer = self.get_best_candidate_layer(mask=mask)
        self.layers = np.append(self.layers, [new_layer], axis=0)

    def fill_layers(self, num_layers: int) -> None:
        """Add layers until there are num_layers.

        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers
        """
        if len(self.layers) >= num_layers:
            return
        try:
            self._add_one_layer()
        except ColorsExhaustedError:
            return
        self.fill_layers(num_layers)

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
            layer_masks = cast("Iterable[_IntA]", self.get_layer_masks())
            self.layers.resize((0, self.layers.shape[1]), refcheck=False)
            for mask in layer_masks:
                self._add_one_layer(mask=mask)
        self.two_pass_fill_layers(num_layers)

    # ===============================================================================
    #   Define and select new candidate layers
    # ===============================================================================

    def _new_candidate_layer(self, palette_index: int) -> _IntA:
        """Create a new candidate layer.

        :param palette_index: the index of the color to use in the new layer
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
        layers = cast("Iterable[npt.NDArray[np.intp]]", self.layers)
        state_cost_matrix = self.target.get_cost_matrix(*layers)
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
        layers = cast("Iterable[npt.NDArray[np.intp]]", self.layers)
        state = merge_layers(*layers)
        state_cost = self.target.get_cost(state, mask=mask)
        available_colors = self.get_available_colors()

        if not available_colors:
            raise ColorsExhaustedError

        candidates = [self._new_candidate_layer(x) for x in available_colors]
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


def _expand_layers(
    quantized_image: Annotated[npt.NDArray[np.intp], "(r, c)"],
    d1_layers: Annotated[npt.NDArray[np.intp], "(n, 512)"],
) -> Annotated[npt.NDArray[np.intp], "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.

    Convert the (usually (512,)) layers of an ImageApproximation to the (n, r, c)
    layers required by draw_posterized_image.
    """
    d1_layers_ = cast("Iterable[npt.NDArray[np.intp]]", d1_layers)
    return np.array([x[quantized_image] for x in d1_layers_])


class Posterization:
    """Result of posterizing an image.

    :param indices: (r, c) array with palette indices from the quantized image
    :param palette: (512, 3) array of color vectors
    :param layers: (n, 512) array of n layers, each containing a value (color index)
        and -1 for transparent
    :param expanded_layers: (n, r, c) array of layers expanded to the size of the
        quantized image
    """

    def __init__(
        self,
        indices: Annotated[npt.NDArray[np.intp], "(r, c)"],
        palette: Annotated[npt.NDArray[np.uint8], "(512, 3)"],
        layers: Annotated[npt.NDArray[np.intp], "(n, 512)"],
    ) -> None:
        """Initialize the Posterization.

        :param indices: (r, c) array with palette indices from the quantized image
        :param palette: (512, 3) array of color vectors
        :param layers: (n, 512) array of n layers, each containing a value (color index)
            and -1 for transparent
        """
        self.indices = indices
        self.palette = palette
        self.layers = layers
        self.expanded_layers = _expand_layers(indices, layers)


def posterize(
    image_path: Path,
    num_cols: int,
    *,
    savings_weight: None | float = None,
    vibrant_weight: None | float = None,
    ignore_quantized_image_cache: bool = False,
    resolution: None | int = None,
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
    if resolution is not None:
        from PIL import Image

        image = Image.open(image_path)
        if max(image.size) > resolution:
            image.thumbnail(
                (resolution, resolution), Image.Resampling.LANCZOS
            )
        target = new_target_image(image, ignore_cache=ignore_quantized_image_cache)
    else:
        target = new_target_image(image_path, ignore_cache=ignore_quantized_image_cache)
    state = ImageApproximation(
        target, savings_weight=savings_weight, vibrant_weight=vibrant_weight
    )
    state.two_pass_fill_layers(num_cols)
    return Posterization(target.indices, target.palette, state.layers)
