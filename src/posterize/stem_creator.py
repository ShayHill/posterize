"""Create a stem for a filename from a list of arguments.

:author: Shay Hill
:created: 2026-02-08
"""

from collections.abc import Iterator
from pathlib import Path


def _percentage_infix(float_: float) -> str:
    """Get a string to use in the filename for a percentage."""
    return f"{int(float_ * 100):02}"


def _iter_stem_parts(*args: Path | float | str | list[int] | None) -> Iterator[str]:
    """Convert args to strings and filter out empty strings.

    This is only suited for args where all floats are between 0 and 1.
    """
    if not args:
        return
    arg, *tail = args
    if arg is None:
        pass
    elif isinstance(arg, str):
        yield arg
    elif isinstance(arg, Path):
        yield arg.stem
    elif isinstance(arg, float):
        assert 0 <= arg <= 1
        yield _percentage_infix(arg)
    elif isinstance(arg, list):
        yield "".join(f"i{x}" for x in arg)
    else:
        assert isinstance(arg, int)
        yield f"{arg:03d}"
    yield from _iter_stem_parts(*tail)


def stemize(*args: Path | float | str | list[int] | None) -> str:
    """Create a stem for a filename from a list of arguments."""
    return "-".join(_iter_stem_parts(*args))
