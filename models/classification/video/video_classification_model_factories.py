from typing import Dict

from torch.nn import Module
from torchvision.models.video import r3d_18, mvit_v2_s, s3d, r2plus1d_18
from pytorchvideo import models

from models.classification.classification_model_factories import register_classification_model
from models.classification.classification_module import ClassificationModule
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from models.classification.video.video_classification_model_provider import VideoClassificationModelProvider, \
    VideoClassificationModelFactory

from utilities.configuration.configuration_reader import ConfigurationReader


def register_video_classification_model(id: str, factory: VideoClassificationModelFactory):
    register_classification_model(VideoClassificationModelProvider(id, factory))


def create_r3d_18(configuration_reader: ConfigurationReader,
                  model_configuration: VideoClassificationModelConfiguration,
                  num_classes: int) -> Module:
    return r3d_18(weights=None, num_classes=num_classes)


def create_s3d(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
               num_classes: int):
    return s3d(weights=None, num_classes=num_classes)


def create_mvit(configuration_reader: ConfigurationReader, model_configuration: VideoClassificationModelConfiguration,
                num_classes: int):
    return models.create_multiscale_vision_transformers(
        spatial_size=(model_configuration.clip.height, model_configuration.clip.width),
        temporal_size=model_configuration.clip.get_frame_count(), head_num_classes=num_classes, depth=8)


def create_mvit_v2_s(configuration_reader: ConfigurationReader,
                     model_configuration: VideoClassificationModelConfiguration, num_classes: int) -> Module:
    return mvit_v2_s(weights=None)


def create_r2dplus(configuration_reader: ConfigurationReader,
                   model_configuration: VideoClassificationModelConfiguration, num_classes: int) -> Module:
    return r2plus1d_18(weights=None, num_classes=num_classes)


def register_all_video_classification_models():
    register_video_classification_model("r3d_18", create_r3d_18)
    register_video_classification_model("mvit", create_mvit)
    register_video_classification_model("s3d", create_s3d)
    register_video_classification_model("r2d_plus", create_r2dplus)
