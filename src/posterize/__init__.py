"""Import functions into the package namespace.

:author: Shay Hill
:created: 2024-05-09
"""

from posterize.main import ImageApproximation, Posterization, posterize
from posterize.quantization import TargetImage, new_target_image

__all__ = [
    "ImageApproximation",
    "Posterization",
    "TargetImage",
    "new_target_image",
    "posterize",
]
