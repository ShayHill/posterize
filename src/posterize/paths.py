"""Link to potrace binary. Create temporary filenames.

:author: Shay Hill
:created: 2023-07-06
"""

from tempfile import TemporaryFile
from pathlib import Path

with TemporaryFile() as f:
    CACHE_DIR = Path(f.name).parent / "cluster_colors_cache"

_PROJECT = Path(__file__, "../../..").resolve()
_BINARIES = _PROJECT / "binaries"
POTRACE = _BINARIES / "potrace.exe"
CACHE = _BINARIES / "cache"

# hold intermediate images used for calculations
WORKING = _BINARIES / "working"

for directory in (_BINARIES, WORKING, CACHE):
    directory.mkdir(parents=True, exist_ok=True)



