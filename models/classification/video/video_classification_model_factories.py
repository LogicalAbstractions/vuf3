from typing import Dict

from torch.nn import Module
from torchvision.models.video import r3d_18

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


def register_all_video_classification_models():
    register_video_classification_model("r3d_18", create_r3d_18)


