"""Constants and metaparameters for the project.

:author: Shay Hill
:created: 2023-07-06
"""

import numpy as np

# typical rgb to gray scalars r * 0.2988 + g * 0.5870 + b * 0.1140 = gray
RGB_CONTRIB_TO_GRAY = np.array((0.2989, 0.5870, 0.1140))


# metaparameter to remove speckles. The minimum of image width and image height *
# this scalar will be the minimum number of pixels to keep an svg path. Paths
# encompassing fewer than this number of pixels will be discarded.
DEFAULT_MIN_SPECKLE_SIZE_SCALAR = 0.01
