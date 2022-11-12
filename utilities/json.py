import json
from pathlib import Path
from typing import Any

from dataclass_wizard import JSONWizard


def write_json_file(path: Path, obj: Any) -> None:
    with path.open("w") as f:
        f.write(to_json_string(obj))


def print_json(obj: Any) -> None:
    print(to_json_string(obj))


def to_json_string(obj: Any) -> str:
    if isinstance(obj, JSONWizard):
        return obj.to_json(indent=4)
    else:
        return json.dumps(obj, indent=4)
