from data.classification.classification_dataset import ClassificationDataset
from data.video.video_clip import VideoClip

VideoClassificationDataset = ClassificationDataset[VideoClip, str]



