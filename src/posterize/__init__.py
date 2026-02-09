"""Import functions into the package namespace.

:author: Shay Hill
:created: 2024-05-09
"""

from posterize.main import extend_posterization, posterize, posterize_mono
from posterize.posterization import Posterization
from posterize.quantization import (
    TargetImage,
    new_target_image,
    new_target_image_mono,
)

__all__ = [
    "Posterization",
    "TargetImage",
    "extend_posterization",
    "new_target_image",
    "new_target_image_mono",
    "posterize",
    "posterize_mono",
]
