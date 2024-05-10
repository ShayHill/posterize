"""A type to store input images and current error.

:author: Shay Hill
:created: 2024-05-09
"""


from posterize.image_arrays import get_image_pixels, write_monochrome_bmp, _write_bitmap_from_array
from posterize.svg_layers import SvgLayers
from svg_ultralight.strings import svg_color_tuple
from pathlib import Path
import numpy.typing as npt
import numpy as np
from PIL import Image
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from svg_ultralight import write_root
from posterize.paths import WORKING, PROJECT
import itertools as it
from basic_colormath import get_delta_e

from svg_ultralight import new_element, update_element


from cluster_colors import KMedSupercluster, get_image_clusters
from cluster_colors.clusters import Member
from cluster_colors.pool_colors import pool_colors
from cluster_colors.cut_colors import cut_colors
from posterize.color_dist import desaturate_by_color


def get_vivid(color: tuple[float, float, float]) -> float:
    """Get the vividness of a color.

    :param color: color to get the vividness of
    :return: the vividness of the color
    """
    r, g, b = color
    return (max(r, g, b) - min(r, g, b))

def get_clusters(colors: npt.NDArray[np.float64]) -> KMedSupercluster:
    """Pool and cut colors.

    :param colors: colors to pool and cut
    :return: pooled and cut colors
    """
    pooled = pool_colors(colors)
    pooled_and_cut = cut_colors(pooled, 512)
    members = Member.new_members(pooled_and_cut)
    return KMedSupercluster(members)


_INKSCAPE = Path(r"C:\Program Files\Inkscape\bin\inkscape")


SCREEN_HEAD = (
    r'<svg xmlns="http://www.w3.org/2000/svg" '
    r'xmlns:xlink="http://www.w3.org/1999/xlink" '
    r'viewBox="0 0 {0} {1}" width="{0}" height="{1}">'
)
SCREEN_TAIL = "</svg>".encode()

# def _get_rgb_pixels(path: Path) -> npt.NDArray[np.uint8]:
#     """Get the RGB pixels of an image.

#     :param path: path to the image
#     :return: RGB pixels of the image
#     """
#     image = Image.open(path)
#     return np.array(image)


class TargetImage:
    """A type to store input images and current error."""

    def __init__(self, path: Path, elements: list[EtreeElement] | None = None) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param lux: number of quantiles to use in error calculation
        """
        self.path = path
        self.elements: list[EtreeElement] = elements or []

        image = Image.open(path)
        self._bhead = SCREEN_HEAD.format(image.width, image.height).encode()

        self._image_grid = np.array(image).astype(float)
        self._state_grid: npt.NDArray[np.float64] | None = None
        self._error_grid: npt.NDArray[np.float64] | None = None
        self._clusters = get_image_clusters(path)

        self.append_background()

    def get_next_color(self, idx: int = 0) -> tuple[float, float, float]:
        """Get the next color to use.

        :return: the next color to use
        """
        self._clusters.split_to_at_most(18)
        if not self.elements:
            return self._clusters.get_rsorted_exemplars()[0]

        prev_element_colors = []
        for element in self.elements:
            as_str = element.get("fill")
            rgb = as_str[4:-1].split(",")
            prev_element_colors.append(tuple(map(float, rgb)))

        def contrast(color):
            return min([get_delta_e(color, prev) for prev in prev_element_colors])

        all_colors = self._clusters.get_rsorted_exemplars()
        all_colors = sorted(all_colors, key=contrast)[::-1]



#         most_vivid = max(all_colors, key=get_vivid)
#         print(f"most vivid: {most_vivid}")
#         all_colors.remove(most_vivid)
#         all_colors.insert(0, most_vivid)
#         # big = all_colors[:4]
#         # sml = all_colors[4:][::1]
#         # colors = list(it.chain(*zip(big, sml[::-1])))
        idx = min(idx, len(self._clusters) - 1)
        # r, g, b = self._clusters.get_rsorted_exemplars()[::-1][idx]
        r, g, b = all_colors[idx]
        return r, g, b

    def append_background(self) -> None:
        """Append the background to the list of elements."""
        bg_color = self.get_next_color()
        self.append(new_element("rect", width="100%", height="100%", fill=f"rgb{bg_color}"))

    @property
    def state_grid(self) -> npt.NDArray[np.float64]:
        """Get the current image.

        :return: the current image
        """
        if self._state_grid is None:
            msg = f"state image not set for {self.path}"
            raise ValueError(msg)
        return self._state_grid

    @property
    def error_grid(self) -> npt.NDArray[np.float64]:
        """Get the current image.

        :return: the current image
        """
        if self._state_grid is None:
            msg = f"state image not set for {self.path}"
            raise ValueError(msg)
        if self._error_grid is None:
            msg = f"error grid not set with state_grid"
            raise RuntimeError(msg)
        return self._error_grid

    def _get_candidate_error_grid(
        self, candidate: EtreeElement | None = None
    ) -> npt.NDArray[np.float64]:
        if candidate is None:
            state_grid = self.state_grid
        else:
            state_grid = self._raster_state(candidate, WORKING / "candidate.png")
        return np.sum((state_grid - self._image_grid) ** 2, axis=2)

    def get_candidate_error(self, candidate: EtreeElement | None = None) -> float:
        return float(np.sum(self._get_candidate_error_grid(candidate)))

    def append(self, element: EtreeElement) -> None:
        """Append an element to the list of elements.

        :param element: element to append
        """
        self.elements.append(element)
        self._state_grid = self._raster_state(element, WORKING / "state.png")
        self._error_grid = self._get_candidate_error_grid()

        ws = self._error_grid[..., np.newaxis]
        needs = np.concatenate([self._image_grid, ws], axis=2).reshape(-1, 4)
        self._clusters = get_clusters(needs)

    def _raster_state(
            self, candidate: EtreeElement, filename: Path
    ) -> npt.NDArray[np.float64]:
        elements = self.elements + ([candidate])
        bstrings = map(etree.tostring, elements)
        root = etree.fromstring(
            self._bhead + b"\n" + b"\n".join(bstrings) + SCREEN_TAIL
        )
        _ = write_root(_INKSCAPE, filename, root, do_png=True)
        image = Image.open(filename)
        return np.array(image).astype(float)[:, :, 0:3]

    def evaluate_next_color(self, idx: int=0) -> tuple[tuple[float, float, float], npt.NDArray[np.float64]]:
        """Evaluate the next color.

        :return: the error of the next color
        """
        next_color = self.get_next_color(idx)
        this_error = self.error_grid # type: ignore
        solid_color = np.full_like(self._image_grid, next_color)
        next_error = np.sum((solid_color - self._image_grid) ** 2, axis=2)
        error_delta = this_error - next_error
        # error_delta[error_delta < 0] = 0
        error_delta -= np.min(error_delta)
        # zeros = np.full_like(error_delta, 0)
        # error_delta = np.max(zeros, error_delta)
        # assert np.min(error_delta) == 0
        error_delta /= np.max(error_delta)
        error_delta *= 255
        # gray_pixels = desaturate_by_color(self._image_grid, next_color)
        # breakpoint()
        return next_color, error_delta


LUX_LEVELS = 24
RECUR = 6



LUX_LIMIT = 0.7

if __name__  == "__main__":
    target = TargetImage(PROJECT / "tests/resources/girl.jpg")


    idx = 0
    for _ in range(40):
        next_color, next_color_per_pixel = target.evaluate_next_color(idx)
        print(f"next color: {next_color}")
        next_color_per_pixel = 255 - next_color_per_pixel.astype(np.uint8) 
        _write_bitmap_from_array(next_color_per_pixel, "temp.bmp")

        layers = SvgLayers("temp.bmp", despeckle=1/50)
        def _scored(time: float):
            candidate = layers(time)
            _ = update_element(candidate, fill=svg_color_tuple(next_color), opacity="0.65")
            return (target.get_candidate_error(candidate), time, candidate)

        print("scoring lux levels")
        scored = {_scored(t) for t in (x/(LUX_LEVELS-1) * LUX_LIMIT for x in range(LUX_LEVELS))}

        print("scoring recursive")
        for _ in range(RECUR):
            parent_a, parent_b = sorted(scored, key=lambda x: x[0])[:2]
            new_time = (parent_a[1] + parent_b[1]) / 2
            scored.add(_scored(new_time))


        layers.close()

        best = min(scored, key=lambda x: x[0])


        # with SvgLayers("temp.bmp") as layers:
        #     candidates = [layers(i/LUX_LEVELS) for i in range(1, LUX_LEVELS+1)]
        # for candidate in candidates:
        #     _ = update_element(candidate, fill=svg_color_tuple(next_color))
        # scored = [(target.get_candidate_error(c), c) for c in candidates]
        # best = min(scored, key=lambda x: x[0])
        if best[0] < target.get_candidate_error():
            print()
            print("    >>> ", next_color, best[1])
            idx = 0
            target.append(best[2])
        else:
            idx += 1
            print(f"raising idx to {idx}")
        if idx >= 18:
            break

    breakpoint()
    


