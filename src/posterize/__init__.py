"""Import functions into the package namespace.

:author: Shay Hill
:created: 2024-05-09
"""

from posterize.main import posterize
from posterize.posterization import (
    Posterization,
    dump_posterization,
    load_posterization,
)
from posterize.quantization import TargetImage, new_target_image

__all__ = [
    "Posterization",
    "TargetImage",
    "dump_posterization",
    "load_posterization",
    "new_target_image",
    "posterize",
]
