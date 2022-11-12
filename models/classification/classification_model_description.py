from abc import ABC
from dataclasses import dataclass
from typing import List

from models.model_description import ModelDescription


@dataclass
class ClassificationModelDescription(ModelDescription, ABC):
    class_ids: List[str]
    metrics: list[dict[str, float]]
    confusion_matrix: list
