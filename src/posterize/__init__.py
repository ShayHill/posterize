"""Import functions into the package namespace.

:author: ShayHill
:created: 2024-05-09
"""

from posterize.image_processing import draw_approximation
from posterize.main import ImageApproximation, posterize

__all__ = ["ImageApproximation", "draw_approximation", "posterize"]
