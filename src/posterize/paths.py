"""Link to potrace binary. Create temporary filenames.

:author: Shay Hill
:created: 2023-07-06
"""

import enum
from pathlib import Path

_PROJECT = Path(__file__, "../../..").resolve()
BINARIES = _PROJECT / "binaries"
POTRACE = BINARIES / "potrace.exe"


class TmpBmpInfix(enum.Enum):
    """Infixes for temporary bmp files."""

    MONOCHROME = "_monochrome"
    SILHOUETTE = "_silhouette"


def get_tmp_bmp_filename(path_to_input: Path | str, infix: TmpBmpInfix) -> str:
    """Create a path to a bitmap temp file.

    :param path_to_input: the filename of an input file, the path to the input file,
        or any string you'd like to use as a filename stem.
    :param infix: add an infix to the bitmap filename
    :returns: path to an existing or to-be-created bitmap file

    Filename will be the input filename stem with a suffix of the infix, e.g.,
    "my_file_silhouette.bmp"
    """
    stem = Path(path_to_input).stem
    return f"{stem}{infix.value}.bmp"


def get_tmp_svg_filename(path_to_input: Path | str, lux: float) -> str:
    """Create a path to an svg temp file.

    :param path_to_input: the filename of an input file, the path to the input file, or
        any string you'd like to use as a filename stem.
    :param lux: the lux passed to potrace to create the svg.
    :returns: path to an existing or to-be-created svg file

    Filename will include the lux value rounded to two digits, e.g.,
    "my_file_0_50.svg"
    """
    stem = Path(path_to_input).stem
    infix = f"{lux:0.2f}".replace(".", "_")
    return f"{stem}_{infix}.svg"
