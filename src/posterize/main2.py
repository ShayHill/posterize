from __future__ import annotations

import dataclasses
import inspect
import logging
import os
from pathlib import Path
from typing import TypeVar

import numpy as np
from lxml.etree import _Element as EtreeElement  # type: ignore
from svg_ultralight import update_element
from svg_ultralight.strings import svg_color_tuple

from posterize.image_arrays import write_bitmap_from_array
from posterize.paths import CACHE, PROJECT, TEMP, WORKING
from posterize.svg_layers import SvgLayers
from posterize.type_target_image import TargetImage

_T = TypeVar("_T")


def _get_sum_solid_error(
    target: TargetImage, color: tuple[int, int, int]
) -> np.float64:
    """Get one float representing the error of a solid color."""
    solid = target.monochrome_like(color)
    return np.sum(target.get_error(solid))


def _replace_background(target: TargetImage, num: int, idx: int = 0) -> None:
    """Replace the default background color with one of the cluster color exemplars.

    :param target: target image to replace the background of
    :param num: number of colors to use
    :param idx: index of the color to use. Will use the largest cluster exemplar by
        default.
    """
    logging.info("replacing default background color")
    cols = target.get_colors(num)
    scored = [(_get_sum_solid_error(target, col), col) for col in cols]
    target.set_background(min(scored)[1])


def _get_svglayers_instance_from_color(
    target: TargetImage, col: tuple[int, int, int]
) -> SvgLayers:
    """Create an svg layers instance from a TargetImage and a color.

    :param target: target image to use
    :param col: color to use
    :return: svg layers instance

    Write an image representing the gradient (white is better) of how much the given
    color would improve the current state. Use this image to create an SvgLayers
    instance.
    """
    bmp_name = TEMP / f"temp_{'-'.join(map(str, col))}.bmp"
    col_improves = target.get_color_cost_delta_bifurcated(col)
    write_bitmap_from_array(col_improves, bmp_name)
    return SvgLayers(bmp_name, despeckle=1 / 50)


def _get_candidate(
    layers: SvgLayers, col: tuple[int, int, int], lux: float, opacity: float
) -> EtreeElement:
    """Create a candidate element.

    :param layers: svg layers instance to use
    :param col: color to use
    :param lux: lux level to use
    :param opacity: opacity level to use
    :return: candidate element
    """
    return update_element(
        layers(lux), fill=svg_color_tuple(col), opacity=f"{opacity:0.2f}"
    )


@dataclasses.dataclass
class ScoredCandidate:
    score: float
    candidate: EtreeElement

    def __lt__(self, other: ScoredCandidate) -> bool:
        return self.score < other.score


class CandidateScorer:
    """Score candidate lux levels for a given color.

    For each lux level, yield and cache a ScoredCandidate instance."""

    def __init__(
        self, target: TargetImage, color: tuple[int, int, int], opacity: float
    ):
        """Initialize a CandidateScorer.

        :param target: target image for calculating error
        :param color: color to use
        :param opacity: opacity level to use for candidates
        """
        self._target = target
        self._color = color
        logging.info(f"scoring {color}")
        self._opacity = opacity
        self._layers = _get_svglayers_instance_from_color(target, color)
        self._scored: dict[float, ScoredCandidate] = {}

    def __lt__(self, other: CandidateScorer) -> bool:
        """Sort by lowest score found in the cache."""
        return min(self._scored) < min(other._scored)

    def _new_candidate(self, lux: float) -> EtreeElement:
        return _get_candidate(self._layers, self._color, lux, self._opacity)

    def score_candidate(self, lux: float) -> ScoredCandidate:
        """Yield a scored candidate tuple for a given lux level."""
        if lux not in self._scored:
            candidate = self._new_candidate(lux)
            score = self._target.get_sum_candidate_cost(candidate)
            self._scored[lux] = ScoredCandidate(score, candidate)
        return self._scored[lux]

    def score_candidates(self, *luxs: float) -> list[ScoredCandidate]:
        """Yield ScoredCandidate instances for each lux level."""
        return [self.score_candidate(lux) for lux in luxs]

    def get_sorted_scored(self) -> list[float]:
        """Return a list of ScoredCandidate instances sorted by score."""
        sorted_items = sorted(self._scored.items(), key=lambda kv: (kv[1], kv[0]))
        return [k for k, _ in sorted_items]

    def find_best(self, tries: int) -> ScoredCandidate:
        """Find the best candidate for a given color.

        :param tries: number of tries to make
        :return: best candidate

        Iteratively find the best candidate by averaging the best and second
        best-scoring lux values. If the average is not better than the second best,
        give up. Additional iterations would just average the same two values.
        """
        return self.score_candidate(0.5)
        _ = self.score_candidates(0)
        _ = self.score_candidates(1)
        for _ in range(tries):
            best_lux, good_lux = self.get_sorted_scored()[:2]
            test_lux = (best_lux + good_lux) / 2
            test = self.score_candidate(test_lux)
            if test > self._scored[good_lux]:
                break
            logging.info(f"trying {test_lux}")
        logging.info(f"completed with {self.get_sorted_scored()[0]}")
        return self._scored[self.get_sorted_scored()[0]]


def _get_infix(arg_val: Path | float) -> str:
    """Format a Path instance, int, or float into a valid filename infix.

    :param arg_val: value to format
    :return: formatted value
    """
    if isinstance(arg_val, Path):
        return arg_val.stem.replace(" ", "_")
    return str(arg_val).replace(".", "p")


def load_target_image(image_path: Path, *args: float | Path) -> TargetImage:
    """Load a cached TargetImage instance or create a new one.

    :param image_path: path to the image
    :param args: additional arguments to use in the cache identifier. If all of these
        are the same, use the cache. If one of these change, create a new instance.
    :return: a TargetImage instance
    """
    cache_identifiers = [_get_infix(x) for x in (image_path, *args)]
    path_to_cache = CACHE / f"{'_'.join(cache_identifiers)}.xml"
    return TargetImage(image_path, path_to_cache=path_to_cache)


def get_posterize_elements(
    image_path: os.PathLike[str],
    num_cols: int,
    num_luxs: int,
    opacity: float,
    max_layers: int,
) -> EtreeElement:
    """Return a list of elements to create a posterized image.

    :param image_path: path to the image
    :param num_cols: number of colors to split the image into before selecting a
        color for the next element (will add full image exemplar to use num_cols + 1
        colors.
    :param num_luxs: number of lux levels to use
    :param opacity: opacity level to use for candidates
    :param max_layers: maximum number of layers to add to the image

    The first element will be a rect element showing the background color. Each
    additional element will be a path element filled with a color that will
    progressively improve the approximation.
    """
    target = load_target_image(Path(image_path), num_cols, num_luxs, opacity)

    if len(target.root) < 1:
        _replace_background(target, num_cols)

    while len(target.root) < max_layers:
        logging.info(f"layer {len(target.root) + 1}")

        cols = target.get_next_colors(num_cols)
        scorers = [CandidateScorer(target, col, opacity) for col in cols]
        for scorer in scorers:
            _ = scorer.score_candidates(*np.linspace(0, 1, num_luxs)[1:-1])
        best_scorer = min(scorers)
        best = best_scorer.find_best(5)
        # if best.score > target.sum_error:
        #     logging.info("no more improvement found")
        #     break
        target.append(best.candidate)
        if not best.candidate[0].attrib["d"]:
            breakpoint()
        target.render_state(WORKING / "state")

    return target.root


if __name__ == "__main__":
    _ = get_posterize_elements(PROJECT / "tests/resources/bird.jpg", 7, 8, 0.85, 40)
