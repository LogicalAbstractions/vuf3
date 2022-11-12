from dataclasses import dataclass
from pathlib import Path

from dataclass_wizard import JSONWizard


@dataclass
class VideoClipDescription(JSONWizard):
    length: float
    width: int
    height: int
    frame_rate: int

    def get_frame_count(self) -> int:
        return int(self.length * self.frame_rate)

