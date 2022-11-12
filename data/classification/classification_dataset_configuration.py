from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from dataclass_wizard import JSONWizard


@dataclass
class ClassificationDatasetConfiguration(JSONWizard, ABC):
    id: ClassVar[str] = "dataset"

    path: str
    test_path: str
    val_split: float
