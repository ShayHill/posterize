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

import sys
from pathlib import Path
from typing import Union, Optional, Iterable
from svg_ultralight import new_svg_root, format_number, write_svg, new_sub_element, write_png_from_svg

from posterize.svg_layers import SvgLayers

INKSCAPE = str(Path(r"C:\Program Files\Inkscape\bin\inkscape"))

def posterize_with_outline(
    input_: Union[Path, str],
    output: Union[Path, str],
    luxs: Iterable[float],
    cols: Iterable[str],
    background: str,
    strokes: Iterable[str],
    stroke_widths: Iterable[float],
    despeckle: Optional[float] = None,
):
    """A posterized effect a stroke around the silhouette."""
    svg_layers = SvgLayers(input_)
    viewbox = {
        "x": 0,
        "y": 0,
        "width": svg_layers.width,
        "height": svg_layers.height,
    }

    # new_svg_root takes trailing-underscore viewbox arguments
    root = new_svg_root(**{k + "_": v for k, v in viewbox.items()})
    _ = new_sub_element(root, "rect", **viewbox, fill=background)

    for lux, col in zip(luxs, cols):
        if lux == 0.0:
            # create a layer for the stroke around the silhouette
            for stroke, stroke_width in zip(strokes, stroke_widths):
                layer = svg_layers(lux)
                layer.attrib["stroke"] = stroke
                layer.attrib["stroke-width"] = format_number(stroke_width)
                root.append(layer)

        # fill in the dark areas
        layer = svg_layers(lux)
        layer.attrib["fill"] = col
        root.append(layer)

    svg_layers.close()
    _ = write_svg(output, root)
    _ = write_png_from_svg(INKSCAPE, output)
    _ = sys.stdout.write(f"wrote {output}\n")

