"""Link to potrace binary. Create temporary filenames.

:author: Shay Hill
:created: 2023-07-06
"""

from pathlib import Path
from tempfile import TemporaryFile

with TemporaryFile() as f:
    CACHE_DIR = Path(f.name).parent / "cluster_colors_cache"

_PROJECT = Path(__file__, "../../..").resolve()
_BINARIES = _PROJECT / "binaries"
POTRACE = _BINARIES / "potrace.exe"

# hold intermediate images used for calculations
WORKING = _BINARIES / "working"

for directory in (_BINARIES, WORKING):
    directory.mkdir(parents=True, exist_ok=True)
