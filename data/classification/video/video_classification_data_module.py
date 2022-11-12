from typing import Dict

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.classification.video.video_classification_preprocessing_dataset import VideoClassificationPreprocessingDataset
from models.model_task import ModelTask


class VideoClassificationDataModule(LightningDataModule):
    def __init__(self,
                 datasets: Dict[ModelTask, VideoClassificationPreprocessingDataset],
                 batch_size: int,
                 num_workers: int = 1):
        super(VideoClassificationDataModule, self).__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample = self.datasets[ModelTask.TRAINING][0]

    def input_shape(self) -> torch.Size:
        return self.sample[0].size()

    def train_dataloader(self):
        return DataLoader(self.datasets[ModelTask.TRAINING],
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets[ModelTask.VALIDATION],
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets[ModelTask.TESTING],
                          batch_size=self.batch_size, num_workers=self.num_workers)

