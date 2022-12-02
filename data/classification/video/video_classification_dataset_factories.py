import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from data.classification.classification_mapping import ClassificationMapping
from data.classification.transforms.indexed_classification_dataset import random_split
from data.classification.video.folder.folder_video_classification_dataset import FolderVideoClassificationDataset
from data.classification.video.video_classification_data_module import VideoClassificationDataModule
from data.classification.video.video_classification_dataset import VideoClassificationDataset
from data.classification.video.video_classification_dataset_configuration import VideoClassificationDatasetConfiguration
from data.classification.video.video_classification_preprocessing_dataset import VideoClassificationPreprocessingDataset
from data.preprocessing.video.video_preprocessing_configuration import VideoPreprocessingConfiguration
from data.preprocessing.video.video_preprocessing_factories import create_image_preprocessor, create_video_preprocessor
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from models.model_task import ModelTask
from models.video.video_model_configuration import VideoModelConfiguration
from training.classification.classification_training_configuration import ClassificationTrainingConfiguration
from training.training_configuration import TrainingConfiguration
from utilities.configuration.configuration_reader import ConfigurationReader
from utilities.json import print_json


def __print_dataset__(dataset: VideoClassificationDataset, task: ModelTask):
    print(str(task))
    print(dataset)


def create_video_classification_datasets(configuration_reader: ConfigurationReader) -> \
        Tuple[Dict[ModelTask, VideoClassificationDataset], Dict[ModelTask, torch.Tensor], ClassificationMapping[str]]:
    configuration: VideoClassificationDatasetConfiguration = configuration_reader.read_object(
        VideoClassificationDatasetConfiguration)

    print("Dataset configuration")
    print_json(configuration)

    training_configuration: TrainingConfiguration = configuration_reader.read_object(TrainingConfiguration)

    train_val_dataset = FolderVideoClassificationDataset(Path(configuration.path), configuration.clip,
                                                         configuration.fix_missing_frames)
    test_dataset = FolderVideoClassificationDataset(Path(configuration.test_path), configuration.clip,
                                                    configuration.fix_missing_frames)

    train_dataset, val_dataset = random_split(train_val_dataset, configuration.val_split)

    mapping = train_dataset.get_mapping()

    datasets: Dict[ModelTask, VideoClassificationDataset] = {
        ModelTask.TRAINING: train_dataset,
        ModelTask.VALIDATION: val_dataset,
        ModelTask.TESTING: test_dataset
    }

    weights: Dict[ModelTask, torch.Tensor] = dict()

    print("Datasets statistics")
    print(mapping)

    for task, dataset in datasets.items():
        __print_dataset__(dataset, task)
        weights[task] = \
            dataset.get_description().get_class_weight_tensor(dataset.get_mapping().class_indices,
                                                              dataset.get_mapping().class_indices_to_class_id)

        if training_configuration.balance_dataset:
            weight_dict = dataset.get_description().get_class_weights(dataset.get_mapping().class_indices,
                                                                      dataset.get_mapping().class_indices_to_class_id)

            print("Class weights for balancing")
            print_json(weight_dict)

    return datasets, weights, mapping


def create_video_classification_datamodule(configuration_reader: ConfigurationReader,
                                           datasets: Dict[ModelTask, VideoClassificationDataset],
                                           mapping: ClassificationMapping[str]) \
        -> VideoClassificationDataModule:
    result_datasets: Dict[ModelTask, VideoClassificationPreprocessingDataset] = dict()

    model_configuration: VideoClassificationModelConfiguration = \
        configuration_reader.read_object(VideoClassificationModelConfiguration)

    preprocessing_configuration: VideoPreprocessingConfiguration = \
        configuration_reader.read_object(VideoPreprocessingConfiguration)

    training_configuration: ClassificationTrainingConfiguration = \
        configuration_reader.read_object(ClassificationTrainingConfiguration)

    for task in datasets.keys():
        source_dataset = datasets[task]

        image_preprocessor = create_image_preprocessor(model_configuration.clip, preprocessing_configuration, task)
        video_preprocessor = create_video_preprocessor(model_configuration.clip, preprocessing_configuration, task)

        result_datasets[task] = VideoClassificationPreprocessingDataset(source_dataset, mapping, image_preprocessor,
                                                                        video_preprocessor)

    return VideoClassificationDataModule(result_datasets, training_configuration.batch_size,
                                         training_configuration.dataloader_workers)
