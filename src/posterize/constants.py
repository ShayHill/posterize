"""Constants for the project.

:author: Shay Hill
:created: 2023-07-06
"""

import numpy as np

# shrink image files to this size (maximum dimension) before triangulating
WORKING_SIZE = 160

# typical rgb to gray scalars r * 0.2988 + g * 0.5870 + b * 0.1140 = gray
RGB_CONTRIB_TO_GRAY = np.array((0.2989, 0.5870, 0.1140))
