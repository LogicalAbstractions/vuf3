from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Dict

import torch
from torch.nn import Module

from models.classification.classification_module import ClassificationModule
from models.model_task import ModelTask
from training.classification.classification_training_configuration import ClassificationTrainingConfiguration
from training.training_configuration import TrainingConfiguration
from utilities.configuration.configuration_reader import ConfigurationReader


class ClassificationModelProvider(ABC):

    def __init__(self, id: str):
        super(ClassificationModelProvider, self).__init__()
        self.id = id

    def create_module(self,
                      configuration_reader: ConfigurationReader,
                      num_classes: int,
                      class_weights: Optional[Dict[ModelTask, torch.tensor]]) -> ClassificationModule:
        training_configuration: ClassificationTrainingConfiguration = configuration_reader.read_object(
            ClassificationTrainingConfiguration)

        actual_class_weights = class_weights if training_configuration.balance_dataset else None

        classifier = self.__create_classifier__(configuration_reader, num_classes)

        return ClassificationModule(classifier, num_classes, actual_class_weights, training_configuration.learning_rate,
                                    training_configuration.reduce_lr_on_plateau)

    @abstractmethod
    def __create_classifier__(self, configuration_reader: ConfigurationReader, num_classes: int) -> Module:
        pass
