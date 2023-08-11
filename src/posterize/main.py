"""Create a posterized-style image from svg paths generated by potrace.

Example usage of the SvgLayers class. Some of the arguments to these functions are
slightly unintuitive.

:param input: The image to be converted to svg paths. This should be a png image with
    a transparent background. removal.ai is a good site to remove the background from
    an image.

:param output: The path to the svg file you want to create.

:param luxs: a list of illumination levels, [0, 1].

    0 is a special case where all opaque pizels are surrounded by svg paths and all
    transparent background is empty.

    As the illumination increases, the area covered by svg paths *decreases*. The
    first try might be linearly spaced illumination levels like the example. If that
    isn't capturing the details you want, try non-linear spacing.

:param cols: ["#ffffff"] Colors to fill the layers of geometry. This is the
    unintuitive part. As the illumination increases, the colors should get *darker*.
    The higher the illumination, the darker an area has to be to remain dark.

:param background: "#ffffff". The color of the background.

:param strokes: ("#ffffff", ), The colors of a stroke or strokes around the
    silhouette (illumination 0 geometry).

:param stroke_widths: the width of the strokes around the silhouette.

:param despeckle: [0, 1] optionally override the default despeckling. The
    higher the value, the less speckling. Too high, and you won't have any geometry
    at all.

:author: Shay Hill
:created: 2023-07-11
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree
from svg_ultralight import (
    new_element,
    new_sub_element,
    new_svg_root,
    update_element,
    write_png_from_svg,
    write_svg,
)

from posterize.svg_layers import SvgLayers

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lxml.etree import _Element as EtreeElement  # type: ignore


_SvgArgs = dict[str, str | float]

# svg arguments needed to black out an element
_BLACKOUT = {
    "fill": "black",
    "stroke": "black",
    "opacity": "1",
    "fill-opacity": "1",
    "stroke-opacity": "1",
}

# show only the silhouette (any non-transparent pixels) when lux is at or below this
_COMPLETE_DARKNESS = 0


def _get_stroke_id(i: int) -> str:
    """Create an id for a presumably stroke-only element.

    :param i: the index of the stroke
    :return: the id

    This creates a unique id for a mask. It is used to identify the mask when masking
    svg geometry.
    """
    return f"stroke{i}"


def _copy_elem(elem: EtreeElement, **kwargs: str | float) -> EtreeElement:
    """Copy an element.

    :param elem: the element to copy
    :param kwargs: element attributes to add or update
    :return: the copy
    """
    copy_of_elem = etree.fromstring(etree.tostring(elem))
    return update_element(copy_of_elem, **kwargs)


def _new_stroke_masks(
    strokes: Sequence[_SvgArgs], bg_elem: EtreeElement, fg_elem: EtreeElement
):
    """Create a mask for each stroke.

    :param strokes: the strokes to mask
    :param bg_elem: the background element (to be shown)
    :param fg_elem: the foreground element (to be hidden)
    :return: the mask elements

    Strokes around the sihouette should be provided widest (by stroke-width) to
    narrowest. Higher strokes mask lower strokes to allow multiple
    semi-transparent strokes without blending.
    """
    masks: list[EtreeElement] = []
    for i, stroke in enumerate(strokes[1:], start=1):
        masks.append(new_element("mask", id_=_get_stroke_id(i)))
        show = _copy_elem(bg_elem, fill="white")
        hide = _copy_elem(fg_elem, **{**stroke, **_BLACKOUT, "transform": "none"})
        masks[-1].extend([show, hide])
    return masks


def posterize_with_outline(
    input_: Path | str,
    output: Path | str,
    luxs: Iterable[float],
    cols: Iterable[str],
    *,
    inkscape: Path | str | None = None,
    background: str | None,
    strokes: Sequence[_SvgArgs] = (),
    despeckle: float | None = None,
):
    """Create a posterized-style image from svg paths generated by potrace.

    :param input_: The image to be converted to svg paths. This should be a png image
        with a transparent background. removal.ai is a good site to remove the
        background from an image.

    :param output: The path to the svg file you want to create.

    :param luxs: a list of illumination levels, [0, 1].

        0 is a special case where all opaque pizels are surrounded by svg paths and
        all transparent background is empty.

        As the illumination increases, the area covered by svg paths *decreases*. The
        first try might be linearly spaced illumination levels like the example. If
        that isn't capturing the details you want, try non-linear spacing.

    :param cols: ["#ffffff"] Colors to fill the layers of geometry. This is the
        unintuitive part. As the illumination increases, the colors should get
        *darker*.  The higher the illumination, the darker an area has to be to
        remain dark.

    :param background: "#ffffff". Optionally give a background color. The anticipated
        use is for tweaking and testing. Presumably, the final svg will have a
        transparent background.

    :param inkscape: Path to an Inkscape executable. If given, the a png will be
        written in addition to an svg. With some versions of Inkscape, this only
        works if passed *without* the .exe extension.

    :param strokes: ({"stroke": "#ffffff", "stroke-width": "2"}), The attributes of a
        stroke or strokes around the silhouette (illumination 0 geometry). The first
        stroke will be in the back, so it should be the widest. Strokes can be
        semi-transparent, because they are masked by subsequent layers. This means
        you can have multiple, semi-transparent strokes and the colors will not
        blend.

    :param despeckle: [0, 1] optionally override the default despeckling. The higher
        the value, the less speckling. Too high, and you won't have any geometry at
        all.
    """
    with SvgLayers(input_, despeckle) as svg_layers:
        elems = [svg_layers(x) for x in luxs]
        width, height = svg_layers.width, svg_layers.height
    bg_elem = new_element("rect", x=0, y=0, width=width, height=height)

    root = new_svg_root(x_=0, y_=0, width_=width, height_=height)

    # define masks for strokes around the silhouette
    defs = new_sub_element(root, "defs")
    defs.extend(_new_stroke_masks(strokes, bg_elem, elems[0]))

    # add background if color provided
    if background:
        root.append(_copy_elem(bg_elem, fill=background))

    # add any strokes around the silhouette
    for next_i, stroke in enumerate(strokes, start=1):
        stroke_elem = _copy_elem(elems[0], **stroke)
        if next_i < len(strokes):
            stroke_elem.attrib["mask"] = f"url(#{_get_stroke_id(next_i)})"
        root.append(stroke_elem)

    # draw the layered svg paths
    for elem, col in zip(elems, cols):
        root.append(update_element(elem, fill=col))

    _ = write_svg(output, root)
    if inkscape and Path(inkscape).exists():
        _ = write_png_from_svg(inkscape, output)
    _ = sys.stdout.write(f"wrote {output}\n")
