from typing import TypeVar, Callable, Tuple, Generic

from data.classification.classification_dataset import T_SampleData, ClassificationDataset
from data.classification.classification_dataset_description import T_ClassId

T_OutSampleData = TypeVar("T_OutSampleData")
T_OutClassId = TypeVar("T_OutClassId")

ClassificationTransform = Callable[[Tuple[T_SampleData, T_ClassId]], Tuple[T_OutSampleData, T_OutClassId]]


class TransformClassificationDataset(Generic[T_SampleData, T_ClassId, T_OutSampleData, T_OutClassId],
                                     ClassificationDataset[T_OutSampleData, T_OutClassId]):
    def __init__(self, source: ClassificationDataset[T_SampleData, T_ClassId],
                 transform_function: ClassificationTransform):
        super(TransformClassificationDataset, self).__init__()

        self.source = source
        self.transform = transform_function

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Tuple[T_OutSampleData, T_OutClassId]:
        return self.transform_function(self.source[index])


def transform(source: ClassificationDataset[T_SampleData, T_ClassId], transform_function: ClassificationTransform) -> \
        ClassificationDataset[T_OutSampleData, T_OutClassId]:
    return TransformClassificationDataset(source, transform_function)
