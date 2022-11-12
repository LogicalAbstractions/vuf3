from pathlib import Path
from typing import List


def get_sub_directories(path: Path) -> List[Path]:
    return [o for o in path.iterdir() if o.is_dir()]


def get_files(path: Path) -> List[Path]:
    return [o for o in path.iterdir() if o.is_file()]
