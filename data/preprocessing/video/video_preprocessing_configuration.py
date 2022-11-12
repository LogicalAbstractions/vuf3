from dataclasses import dataclass
from typing import ClassVar

from dataclass_wizard import JSONWizard

@dataclass
class VideoPreprocessingConfiguration(JSONWizard):
    id: ClassVar[str] = "preprocessing"