from typing import Callable, Optional, Tuple

import torch
from torchvision.transforms.functional import to_tensor

from data.classification.classification_dataset import ClassificationDataset
from data.classification.classification_mapping import ClassificationMapping
from data.classification.video.video_classification_dataset import VideoClassificationDataset
from data.preprocessing.video.video_preprocessing_factories import VideoTensorTransform


class VideoClassificationPreprocessingDataset(ClassificationDataset[torch.Tensor, int]):

    def __init__(self,
                 source: VideoClassificationDataset,
                 mapping: ClassificationMapping[str],
                 image_preprocessor: Optional[VideoTensorTransform],
                 video_preprocessor: Optional[VideoTensorTransform]):
        super(VideoClassificationPreprocessingDataset, self).__init__()
        self.source = source
        self.mapping = mapping

        self.image_preprocessor = image_preprocessor if not None else lambda x: x
        self.video_preprocessor = video_preprocessor if not None else lambda x: x

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample, label = self.source[index]
        mapped_label: int = self.mapping.get_class_index(label)

        frames = list()
        for frame in sample.get_frames():
            frame_tensor = to_tensor(frame)
            frame_tensor = self.image_preprocessor(frame_tensor)
            frames.append(frame_tensor)

        video_tensor = torch.stack(frames)
        video_tensor = self.video_preprocessor(video_tensor)

        return video_tensor, mapped_label
