from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import ClassVar, List

from dataclass_wizard import JSONWizard


@dataclass
class ModelConfiguration(JSONWizard, ABC):
    id: ClassVar[str] = "model"

    base_id: str

    @abstractmethod
    def get_input_size(self, include_batch: bool) -> List[int]:
        pass
