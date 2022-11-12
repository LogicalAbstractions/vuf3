from typing import Callable

import torch
from torchvision.transforms._presets import VideoClassification

from data.preprocessing.video.video_preprocessing_configuration import VideoPreprocessingConfiguration
from data.video.video_clip_description import VideoClipDescription
from models.model_task import ModelTask
from models.video.video_model_configuration import VideoModelConfiguration

VideoTensorTransform = Callable[[torch.Tensor], torch.Tensor]


def create_image_preprocessor(clip_description: VideoClipDescription,
                              preprocessing_configuration: VideoPreprocessingConfiguration,
                              task: ModelTask) -> VideoTensorTransform:
    return lambda x: x


def create_video_preprocessor(clip_description: VideoClipDescription,
                              preprocessing_configuration: VideoPreprocessingConfiguration,
                              task: ModelTask) -> VideoTensorTransform:
    return VideoClassification(crop_size=(clip_description.height,clip_description.width),
                               resize_size=(clip_description.height, clip_description.width))
