"""Run the posterization functions to create output.

:author: Shay Hill
:created: 2025-11-16
"""


from pathlib import Path
import numpy as np

import itertools as it
from basic_colormath import get_delta_e_matrix, rgb_to_hex, mix_hex, get_delta_e_matrix_hex

from posterize import posterize, draw_approximation
import svg_ultralight as su
INKSCAPE = str(Path(r"C:\Program Files\Inkscape\bin\inkscape"))


PROJECT = Path(__file__).parents[2]
RESOURCES = PROJECT / "resources"

OUT = Path("chaucer.svg")



def _svg_rgb_to_hex(svg_rgb: str) -> str:
    """Convert an SVG rgb string 'rgb(100,255,34)' to a hex string.
    
    :param svg_rgb: SVG rgb string, e.g., 'rgb(100,255,34)'
    :return: Hex string, e.g., '#64ff22'
    """
    rgb_values = svg_rgb.strip("rgb()").split(",")
    r, g, b = (int(value) for value in rgb_values)
    return rgb_to_hex((r, g, b))


def swap_palette(source_img: Path, svg: Path, palette: list[str]) -> str:
    image_approximation = posterize(RESOURCES / "chaucer.webp", len(palette))
    _ = draw_approximation(OUT, image_approximation)
    svg_blem = su.parse_bound_element(svg)
    input_cols: list[str] = []
    for sub in svg_blem.elem:
        as_hex = _svg_rgb_to_hex(sub.attrib['fill'])
        input_cols.append(as_hex)
        sub.set('fill', as_hex)

    dist_map: dict[str, str] = {}

    palette_ = palette[:]
    while(palette_):
        dist_mat = get_delta_e_matrix_hex(input_cols, palette_)
        min_index_flat = dist_mat.argmin()
        min_row, min_col = np.unravel_index(min_index_flat, dist_mat.shape)
        dist_map[input_cols.pop(min_row)] = palette_.pop(min_col)

    for sub in svg_blem.elem:
        current_col = sub.attrib['fill']
        sub.set('fill', dist_map[current_col])

    root = su.new_svg_root_around_bounds(svg_blem)
    root.append(svg_blem.elem)
    _ = su.write_svg(svg, root)
    _ = su.write_png_from_svg(INKSCAPE, svg)



palette = ["#c00000", "#1155cc", "#ffc740", "#000000", "#ffffff", "#0e7653"]

# for t in np.linspace(0.5, 0.75, 3):
for t in (0.75,):
    for i in range(2):
        palette.append(mix_hex(palette[i], "#ffffff", ratio=t))
    for i in range(2):
        palette.append(mix_hex(palette[i], "#000000", ratio=t))
palette.append(mix_hex(palette[2], "#ffffff", ratio=0.5))

for i, j in it.combinations(palette[:3], 2):
    palette.append(mix_hex(i, j, ratio=0.5))
    # palette.append(mix_hex(i, j, ratio=0.25))
    # palette.append(mix_hex(i, j, ratio=0.75))

    # palette.append(mix_hex(palette[2], "#ffffff", ratio=0.75))
   
swap_palette(RESOURCES / "chaucer.webp", OUT, palette)

blem = su.parse_bound_element(OUT)
# breakpoint()






