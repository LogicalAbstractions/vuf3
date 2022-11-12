from typing import Type, Tuple

import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelPruning, \
    QuantizationAwareTraining, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from data.classification.classification_mapping import ClassificationMapping
from models.classification.classification_model_configuration import ClassificationModelConfiguration
from models.classification.classification_module import ClassificationModule
from models.export.onnx_export import export_classification_model_to_onnx
from training.classification.classification_training_configuration import ClassificationTrainingConfiguration
from utilities.configuration.configuration_reader import ConfigurationReader
from utilities.json import print_json
from utilities.tables import confusion_matrix_to_table


def test_classification_model(trainer: Trainer, model: ClassificationModule, data_module: LightningDataModule,
                              mapping: ClassificationMapping) -> Tuple[list[dict[str, float]], Tensor]:
    test_metrics = trainer.test(model=model, datamodule=data_module)

    print("Confusion")
    print(confusion_matrix_to_table(mapping, model.test_confusion_matrix_accumulator))

    return test_metrics, model.test_confusion_matrix_accumulator


def run_classification_training(configuration_reader: ConfigurationReader,
                                model_configuration_type: Type,
                                data_module: LightningDataModule,
                                model: ClassificationModule,
                                mapping: ClassificationMapping) -> Tuple[list[dict[str, float]], Tensor]:
    training_configuration: ClassificationTrainingConfiguration = configuration_reader.read_object(
        ClassificationTrainingConfiguration)

    print("Training configuration")
    print_json(training_configuration)

    model_configuration: ClassificationModelConfiguration = configuration_reader.read_object(model_configuration_type)

    logger = True

    if training_configuration.wandb_project is not None:
        print("Enabling wandb")
        logger = WandbLogger(
            save_dir=str(configuration_reader.get_artifact_path() / "wandb"),
            project=training_configuration.wandb_project)
        logger.experiment.config.update(model_configuration.to_dict())
        logger.experiment.config.update(training_configuration.to_dict())

    callbacks = list()

    callbacks.append(ModelCheckpoint(save_last=True, monitor="val_loss", mode="min", save_weights_only=True))

    if training_configuration.pruning is not None and len(training_configuration.pruning) > 0:
        print("Enabling model pruning: " + training_configuration.pruning + ", amount: " + str(
            training_configuration.pruning_amount))
        callbacks.append(ModelPruning(training_configuration.pruning, amount=training_configuration.pruning_amount))

    if training_configuration.early_stopping > 0:
        print("Enabling early stopping: " + str(training_configuration.early_stopping))
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=training_configuration.early_stopping))

    if training_configuration.swa_lrs > 0.0:
        print("Enabling stochastic weight averaging: " + str(training_configuration.swa_lrs))
        callbacks.append(StochasticWeightAveraging(swa_lrs=training_configuration.swa_lrs))

    if training_configuration.quantization_aware:
        print("Enabling quantization aware training, keeping inputs: " + str(
            training_configuration.quantization_keep_inputs))
        callbacks.append(QuantizationAwareTraining(input_compatible=training_configuration.quantization_keep_inputs))

    gradient_clip_val = \
        training_configuration.gradient_clipping if training_configuration.gradient_clipping > 0 else None

    if gradient_clip_val is not None:
        print("Enabling gradient clipping: " + str(gradient_clip_val))

    trainer = Trainer(logger=logger,
                      max_epochs=training_configuration.epochs,
                      default_root_dir=str(configuration_reader.get_artifact_path()),
                      accelerator=training_configuration.accelerator,
                      devices=training_configuration.device_count,
                      precision=training_configuration.precision,
                      auto_lr_find=training_configuration.auto_find_learning_rate,
                      gradient_clip_val=gradient_clip_val,
                      accumulate_grad_batches=training_configuration.accumulate_batches,
                      callbacks=callbacks
                      )

    if training_configuration.auto_find_learning_rate:
        print("Trying to find learning rate")
        trainer.tune(model=model, datamodule=data_module)

    trainer.fit(model=model, datamodule=data_module)

    print("Last model")
    test_classification_model(trainer, model, data_module, mapping)

    print("Best model at: " + trainer.checkpoint_callback.best_model_path)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    test_metrics, confusion_matrix = test_classification_model(trainer, model, data_module, mapping)

    onnx_path = configuration_reader.get_artifact_path() / "model.onnx"

    export_classification_model_to_onnx(onnx_path, model, model_configuration)

    return test_metrics, confusion_matrix
