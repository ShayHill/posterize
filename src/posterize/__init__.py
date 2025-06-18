"""Import functions into the package namespace.

:author: ShayHill
:created: 2024-05-09
"""

from posterize.image_processing import draw_approximation
from posterize.main import ImageApproximation, posterize
from posterize.quantization import TargetImage, new_target_image

__all__ = [
    "ImageApproximation",
    "TargetImage",
    "draw_approximation",
    "new_target_image",
    "posterize",
]
