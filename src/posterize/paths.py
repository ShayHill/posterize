"""Links to binaries and directories.

:author: Shay Hill
:created: 2023-07-06
"""

from typing import Union, Optional
from pathlib import Path
import re

_PROJECT = Path(__file__, "../../..").resolve()
_BINARIES = _PROJECT / "binaries"

POTRACE = _BINARIES / "potrace.exe"
BITMAPS = _BINARIES / "bmps"
SVGS = _BINARIES / "svgs"

# create folders if they do not exist

for folder in [BITMAPS, SVGS]:
    if not folder.exists():
        folder.mkdir()


def get_bmp_path(filename: Union[Path, str], infix: Optional[str] = None) -> Path:
    """Create a path to a bitmap temp file.

    :param filename: the filename of an input file, the path to the input file, or
        any string you'd like to use as a filename stem.
    :param infix: optionally add an infix to the filename stem
        e.g., stem_infix.bmp
    :returns: path to an existing or to-be-created bitmap file
    """
    stem = Path(filename).stem
    if infix:
        stem += f"_{infix}"
    return (BITMAPS / stem).with_suffix(".bmp")


def get_svg_path(filename: Union[Path, str], blacklevel: float) -> Path:
    """Create a path to an svg temp file.

    :param filename: the filename of an input file, the path to the input file, or
        any string you'd like to use as a filename stem.
    :param blacklevel: the blacklevel passed to potrace to create the svg.
    :returns: path to an existing or to-be-created svg file
    """
    stem = Path(filename).stem
    infix = f"{blacklevel:0.2f}".replace(".", "_")
    return (SVGS / f"{stem}_{infix}").with_suffix(".svg")


def clear_temp_files(filename: Optional[Union[Path, str]] = None) -> None:
    """Delete all (or some) temp files.

    :param filename: optionally, delete only temp files associated with this file.

    You can do this to save disk space, but the real reason is to reset if you update
    the algorithm. The filename patterns aren't air tight, and you can end up
    deleting temporary files for "wagoneer" when deleting temporart files for
    "wagon".
    """
    if filename:
        stem = Path(filename).stem
        pattern_bmp = re.compile(f"{stem}.*bmp")
        pattern_svg = re.compile(f"{stem}.*svg")
    else:
        pattern_bmp = re.compile(".*bmp")
        pattern_svg = re.compile(".*svg")

    for file in BITMAPS.glob("*"):
        if pattern_bmp.match(file.name):
            file.unlink()
    for file in SVGS.glob("*"):
        if pattern_svg.match(file.name):
            file.unlink()
