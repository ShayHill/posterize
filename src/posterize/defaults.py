"""Default values for posterization.

:author: Shay Hill
:created: 2026-02-08
"""

# Default weight for sum savings vs. average savings. Average savings is, by default,
# weighted highly. These values are used when selecting the best candidate for the
# next layer color. A higher average savings weight means colors that improve the
# approximation a lot in a small area are chosen over colors that improve the
# approximation a tiny amount over a large area. _DEFAULT_SAVINGS_WEIGHT is the
# weight given to sum savings. (1 - _DEFAULT_SAVINGS_WEIGHT) is the weight given to
# average savings.
SAVINGS_WEIGHT = 0.25


# A higher number (1.0 is maximum) means colors that are more vibrant are more likely
# to be selected as layer colors. The default is 0.0, which will be good for most
# images, but the parameter is available if you have an overall drab image with a few
# bright highlights and want to pay less attention to the background.
VIBRANT_WEIGHT = 0.0


# Resize images larger than this to this maximum dimension. This value is necessary,
# because you can only create arrays of a certain size. A smaller value might speed
# up testing, but the quantization cache will need to be cleared if this value
# changes.
MAX_DIM = 500
