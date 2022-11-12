from dataclasses import dataclass
from typing import ClassVar

from dataclass_wizard import JSONWizard

from data.classification.classification_dataset_configuration import ClassificationDatasetConfiguration
from data.video.video_clip_description import VideoClipDescription


@dataclass
class VideoClassificationDatasetConfiguration(ClassificationDatasetConfiguration):

    fix_missing_frames: bool
    clip: VideoClipDescription
