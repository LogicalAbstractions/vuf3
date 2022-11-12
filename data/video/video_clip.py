from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image


class VideoClip(ABC):

    def __init__(self):
        super(VideoClip, self).__init__()

    @abstractmethod
    def get_frames(self) -> List[Image]:
        pass
