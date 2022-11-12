import random
from typing import Generic, Tuple, List

from torch.utils.data.dataset import T_co

from data.classification.classification_dataset import T_SampleData, ClassificationDataset
from data.classification.classification_dataset_description import T_ClassId


class IndexedClassificationDataset(Generic[T_SampleData, T_ClassId], ClassificationDataset[T_SampleData, T_ClassId]):
    def __init__(self, source: ClassificationDataset[T_SampleData, T_ClassId], indices: List[int]):
        super(IndexedClassificationDataset, self).__init__()

        self.source = source
        self.indices = indices

    def __getitem__(self, index: int) -> T_co:
        return self.source[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(source: ClassificationDataset[T_SampleData, T_ClassId], split_point: float = 0.5) -> \
        Tuple[ClassificationDataset[T_SampleData, T_ClassId], ClassificationDataset[T_SampleData, T_ClassId]]:
    indices = [i for i in range(0, len(source))]
    random.shuffle(indices)
    split_index = int(1.0 - float(len(source)) * split_point)

    a = IndexedClassificationDataset(source, indices[:split_index])
    b = IndexedClassificationDataset(source, indices[split_index:])

    return a, b
