from typing import Callable

from torch.nn import Module

from models.classification.classification_model_provider import ClassificationModelProvider
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from utilities.configuration.configuration_reader import ConfigurationReader

VideoClassificationModelFactory = Callable[[ConfigurationReader, VideoClassificationModelConfiguration, int], Module]


class VideoClassificationModelProvider(ClassificationModelProvider):
    def __init__(self, id: str,
                 factory: VideoClassificationModelFactory):
        super(VideoClassificationModelProvider, self).__init__(id)
        self.factory = factory

    def create_classifier(self, configuration_reader: ConfigurationReader, num_classes: int) -> Module:
        model_configuration: VideoClassificationModelConfiguration = configuration_reader.read_object(
            VideoClassificationModelConfiguration)

        return self.__create_video_classifier__(configuration_reader, model_configuration, num_classes)

    def __create_video_classifier__(self, configuration_reader: ConfigurationReader,
                                    model_configuration: VideoClassificationModelConfiguration,
                                    num_classes: int) -> Module:
        return self.factory(configuration_reader, model_configuration, num_classes)
