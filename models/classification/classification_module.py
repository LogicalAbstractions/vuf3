from pathlib import Path
from typing import Any, Optional, Dict

import onnx
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.nn import Module
from torch.optim import Adam
from torchmetrics import ConfusionMatrix

from models.classification.classification_metric_factories import create_classification_metrics
from models.model_configuration import ModelConfiguration
from models.model_task import ModelTask
from utilities.tables import confusion_matrix_to_table


class ClassificationModule(LightningModule):
    def __init__(self,
                 classifier: Module,
                 num_classes: int,
                 class_weights: Optional[Dict[ModelTask, torch.Tensor]] = None,
                 learning_rate: float = 1e-03, reduce_lr_on_plateau: bool = False):
        super(ClassificationModule, self).__init__()

        self.classifier = classifier
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        self.save_hyperparameters()
        self.metrics = create_classification_metrics(num_classes)
        self.test_confusion_matrix = ConfusionMatrix(num_classes)
        self.test_confusion_matrix_accumulator: Optional[torch.Tensor] = None

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        return self.__default_step__(batch, batch_idx, ModelTask.TRAINING)

    def validation_step(self, batch, batch_idx):
        return self.__default_step__(batch, batch_idx, ModelTask.VALIDATION)

    def test_step(self, batch, batch_idx):
        loss = self.__default_step__(batch, batch_idx, ModelTask.TESTING)

        return loss

    def predict_step(self, batch, batch_idx, **kwargs) -> Any:
        return self.classifier(batch)

    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, monitor="val_loss")

        if self.reduce_lr_on_plateau:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            return optimizer

    def on_test_start(self) -> None:
        super(ClassificationModule, self).on_test_start()
        self.test_confusion_matrix_accumulator = None

    def on_test_end(self) -> None:
        super(ClassificationModule, self).on_test_end()

    def export_onnx(self, model_configuration: ModelConfiguration, path: Path):
        input_size = model_configuration.get_input_size(True)
        dummy_input = torch.randn(input_size, requires_grad=True)
        self.to_onnx(path, dummy_input,
                     input_names=["input"],
                     output_names=["output"],
                     export_params=True,
                     opset_version=17,
                     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

        exported_model = onnx.load(str(path))
        onnx.checker.check_model(exported_model, True)

    def __default_step__(self, batch, batch_idx, task: ModelTask):
        video, label = batch
        prediction = self.classifier(video)
        loss = torch.nn.functional.cross_entropy(prediction, label, weight=self.__get_class_weights__(task))

        self.log(f"{task.value}_loss", loss, on_epoch=True, logger=True, prog_bar=True)
        metrics_result = self.metrics[task.value + "_metrics"](prediction, label)
        self.log_dict(metrics_result)

        if task == ModelTask.TESTING:
            confusion_matrix = self.test_confusion_matrix(prediction, label)
            if self.test_confusion_matrix_accumulator is None:
                self.test_confusion_matrix_accumulator = confusion_matrix
            else:
                self.test_confusion_matrix_accumulator = torch.add(self.test_confusion_matrix_accumulator,
                                                                   confusion_matrix)

        return loss

    def __get_class_weights__(self, task: ModelTask) -> Optional[torch.Tensor]:
        if self.class_weights is not None:
            tensor = self.class_weights[task]
            return tensor.to(self.device)
        return None
