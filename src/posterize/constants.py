"""Constants and metaparameters for the project.

:author: Shay Hill
:created: 2023-07-06
"""

# metaparameter to remove speckles. The minimum of image width and image height *
# this scalar will be the minimum number of pixels to keep an svg path. Paths
# encompassing fewer than this number of pixels will be discarded.
DEFAULT_MIN_SPECKLE_SIZE_SCALAR = 0.01
