from dataclasses import dataclass

from training.training_configuration import TrainingConfiguration


@dataclass
class ClassificationTrainingConfiguration(TrainingConfiguration):
    balance_dataset: bool = True
