from torch.nn import ModuleDict
from torchmetrics import MetricCollection, Accuracy

from data.classification.classification_mapping import ClassificationMapping
from models.model_task import ModelTask


def __create_classification_metrics_for_task__(task: ModelTask, num_classes: int) -> MetricCollection:
    return MetricCollection({
        "accuracy": Accuracy(num_classes=num_classes),
        "top2_accuracy": Accuracy(num_classes=num_classes, top_k=2),
        "top3_accuracy": Accuracy(num_classes=num_classes, top_k=3)
    }, prefix=str(task.value) + "_")


def create_classification_metrics(num_classes: int) -> ModuleDict:
    result = ModuleDict({
        ModelTask.TRAINING.value + "_metrics": __create_classification_metrics_for_task__(ModelTask.TRAINING, num_classes),
        ModelTask.VALIDATION.value + "_metrics": __create_classification_metrics_for_task__(ModelTask.VALIDATION, num_classes),
        ModelTask.TESTING.value + "_metrics": __create_classification_metrics_for_task__(ModelTask.TESTING, num_classes)
    })

    return result
