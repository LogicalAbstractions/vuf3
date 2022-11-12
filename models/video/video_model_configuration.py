from abc import ABC
from dataclasses import dataclass
from typing import List

from data.video.video_clip_description import VideoClipDescription
from models.model_configuration import ModelConfiguration


@dataclass
class VideoModelConfiguration(ModelConfiguration, ABC):
    clip: VideoClipDescription

    def get_input_size(self, include_batch: bool) -> List[int]:
        if include_batch:
            return [1, 3, self.clip.get_frame_count(), self.clip.height, self.clip.width]
        else:
            return [3, self.clip.get_frame_count(), self.clip.height, self.clip.width]
