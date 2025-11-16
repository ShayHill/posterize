"""See full diffs in pytest.

:author: Shay Hill
:created: 2023-10-07
"""

from typing import Any


def pytest_assertrepr_compare(
    config: Any, op: str, left: str, right: str
) -> list[str] | None:
    """See full error diffs"""
    del config
    if op in ("==", "!="):
        return [f"{left} {op} {right}"]
    return None
