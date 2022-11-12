from abc import ABC
from dataclasses import dataclass
from typing import List

from dataclass_wizard import JSONWizard


@dataclass
class ModelDescription(JSONWizard, ABC):
    input_size: List[int]
