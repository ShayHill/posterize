"""Link to potrace binary. Create temporary filenames.

:author: Shay Hill
:created: 2023-07-06
"""

from pathlib import Path
from tempfile import TemporaryFile

with TemporaryFile() as f:
    CACHE_DIR = Path(f.name).parent / "cluster_colors_cache"
