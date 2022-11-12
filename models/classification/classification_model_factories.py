import json
from typing import Dict, Optional, Type

import torch

from data.classification.classification_mapping import ClassificationMapping
from models.classification.classification_model_configuration import ClassificationModelConfiguration
from models.classification.classification_model_description import ClassificationModelDescription
from models.classification.classification_model_provider import ClassificationModelProvider
from models.classification.classification_module import ClassificationModule
from models.model_task import ModelTask
from utilities.configuration.configuration_reader import ConfigurationReader

__providers__: Dict[str, ClassificationModelProvider] = dict()

from utilities.json import print_json


def register_classification_model(provider: ClassificationModelProvider) -> None:
    __providers__[provider.id] = provider


def create_classification_module(configuration_reader: ConfigurationReader,
                                 model_configuration_type: Type,
                                 num_classes: int,
                                 class_weights: Optional[Dict[ModelTask, torch.Tensor]]) -> ClassificationModule:
    model_configuration: ClassificationModelConfiguration = configuration_reader.read_object(
        model_configuration_type)

    print("Model configuration")
    print_json(model_configuration)

    base_id = model_configuration.base_id
    print(f"Trying to create base model: {base_id}")

    model_provider = __providers__[base_id]

    if model_provider is None:
        raise ModuleNotFoundError(name=base_id)

    return model_provider.create_module(configuration_reader, num_classes, class_weights)



