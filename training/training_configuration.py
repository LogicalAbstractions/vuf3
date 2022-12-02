from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from dataclass_wizard import JSONWizard


@dataclass
class TrainingConfiguration(JSONWizard, ABC):
    id: ClassVar[str] = "training"
    balance_dataset: bool = False
    batch_size: int = 1
    accumulate_batches: int = 1
    epochs: int = 100
    learning_rate: float = 1.003
    auto_find_learning_rate: bool = False
    reduce_lr_on_plateau: bool = False
    early_stopping: int = 0
    swa_lrs: float = -1.0
    pruning: str = "l1_unstructured"
    pruning_amount: float = 0.5
    quantization_aware: bool = False
    quantization_keep_inputs: bool = False
    gradient_clipping: float = 0.0
    dataloader_workers: int = 4
    device_count: int = -1
    accelerator: str = "gpu"
    precision: int = 32
    wandb_project: str = "vuf3"
