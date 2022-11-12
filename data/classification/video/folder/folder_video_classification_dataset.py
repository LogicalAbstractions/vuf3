from pathlib import Path
from typing import List, Tuple

from data.classification.video.folder.folder_video_clip import FolderVideoClip
from data.classification.video.video_classification_dataset import VideoClassificationDataset
from data.video.video_clip import VideoClip
from data.video.video_clip_description import VideoClipDescription
from utilities.paths import get_sub_directories, get_files


class FolderVideoClassificationDataset(VideoClassificationDataset):

    def __init__(self, path: Path, clip_description: VideoClipDescription,
                 fix_missing_frames: bool = True):
        super(FolderVideoClassificationDataset, self).__init__()

        self.path = path
        self.fix_missing_frames = fix_missing_frames
        self.clip_description = clip_description
        self.frame_count = clip_description.get_frame_count()

        self.samples: List[FolderVideoClip] = []

        class_paths = [o for o in path.iterdir() if o.is_dir()]

        for class_path in class_paths:
            sample_paths = get_sub_directories(class_path)

            for sample_path in sample_paths:
                file_count = len(get_files(sample_path))

                self.samples.append(FolderVideoClip(sample_path, sample_path.parent.name.lower(), self.frame_count,
                                                    self.fix_missing_frames))
                if file_count != self.frame_count:
                    print("Invalid sample found: {}/{} frames at {}, fixing by duplicating last frame"
                          .format(file_count, self.frame_count, sample_path))

    def __getitem__(self, index: int) -> Tuple[VideoClip, str]:
        return self.samples[index], self.samples[index].class_id

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"Path: {self.path}\n" + super(FolderVideoClassificationDataset, self).__str__()
