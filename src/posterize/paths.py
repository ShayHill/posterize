"""Link to potrace binary. Create temporary filenames.

:author: Shay Hill
:created: 2023-07-06
"""

import enum
from pathlib import Path
from typing import Union

_PROJECT = Path(__file__, "../../..").resolve()
BINARIES = _PROJECT / "binaries"
POTRACE = BINARIES / "potrace.exe"


class TempBmpInfix(enum.Enum):
    """Infixes for temporary bmp files."""

    MONOCHROME = "_monochrome"
    SILHOUETTE = "_silhouette"


def get_temp_bmp_filename(input_filename: Union[Path, str], infix: TempBmpInfix) -> str:
    """Create a path to a bitmap temp file.

    :param input_filename: the filename of an input file, the path to the input file,
        or any string you'd like to use as a filename stem.
    :param infix: add an infix to the bitmap filename
    :returns: path to an existing or to-be-created bitmap file

    Filename will be the input filename stem with a suffix of the infix, e.g.,
    "my_file_silhouette.bmp"
    """
    stem = Path(input_filename).stem
    return f"{stem}{infix.value}.bmp"


def get_temp_svg_filename(filename: Union[Path, str], illumination: float) -> str:
    """Create a path to an svg temp file.

    :param filename: the filename of an input file, the path to the input file, or
        any string you'd like to use as a filename stem.
    :param illumination: the illumination passed to potrace to create the svg.
    :returns: path to an existing or to-be-created svg file

    Filename will include the illumination value rounded to two digits, e.g.,
    "my_file_0_50.svg"
    """
    stem = Path(filename).stem
    infix = f"{illumination:0.2f}".replace(".", "_")
    return f"{stem}_{infix}.svg"
