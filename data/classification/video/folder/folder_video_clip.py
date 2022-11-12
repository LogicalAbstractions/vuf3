from pathlib import Path
from typing import List

from PIL import Image

from data.video.video_clip import VideoClip
from utilities.paths import get_files


def __load_image__(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    image = image._new(image.im)
    return image


class FolderVideoClip(VideoClip):
    def __init__(self, folder_path: Path, class_id: str, frame_count: int, fix_missing_frames: bool):
        super(FolderVideoClip, self).__init__()

        self.folder_path = folder_path
        self.class_id = class_id
        self.frame_count = frame_count
        self.fix_missing_frames = fix_missing_frames

    def get_frames(self) -> List[Image.Image]:
        file_paths = get_files(self.folder_path)
        sorted_file_paths = sorted(file_paths, key=lambda f: int(f.with_suffix('').name))

        if len(sorted_file_paths) > 0:
            last_frame_path = sorted_file_paths[len(sorted_file_paths) - 1]
            if self.fix_missing_frames:
                while len(sorted_file_paths) < self.frame_count:
                    sorted_file_paths.append(last_frame_path)

        return [__load_image__(o) for o in sorted_file_paths]
