from abc import ABC
from dataclasses import dataclass

from models.model_configuration import ModelConfiguration


@dataclass
class ClassificationModelConfiguration(ModelConfiguration, ABC):
    pass