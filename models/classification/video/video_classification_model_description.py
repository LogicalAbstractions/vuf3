from dataclasses import dataclass

import torch

from data.classification.classification_mapping import ClassificationMapping
from data.video.video_clip_description import VideoClipDescription
from models.classification.classification_model_description import ClassificationModelDescription
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from utilities.configuration.configuration_reader import ConfigurationReader
from utilities.tables import confusion_matrix_to_table, table_to_json_data


@dataclass
class VideoClassificationModelDescription(ClassificationModelDescription):
    clip: VideoClipDescription

    @staticmethod
    def create(configuration_reader: ConfigurationReader,
               mapping: ClassificationMapping,
               metrics: list[dict[str, float]],
               confusion_matrix: torch.Tensor) -> 'VideoClassificationModelDescription':
        model_configuration: VideoClassificationModelConfiguration = configuration_reader.read_object(
            VideoClassificationModelConfiguration)

        confusion_table = confusion_matrix_to_table(mapping, confusion_matrix)

        return VideoClassificationModelDescription(model_configuration.get_input_size(True), mapping.class_ids, metrics,
                                                   table_to_json_data(confusion_table),
                                                   model_configuration.clip)
