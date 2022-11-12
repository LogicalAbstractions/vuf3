from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Tuple, Optional, Dict, Set

from prettytable import PrettyTable
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from data.classification.classification_dataset_description import ClassificationDatasetDescription, T_ClassId
from data.classification.classification_mapping import ClassificationMapping
from utilities.tables import add_sub_table

T_SampleData = TypeVar("T_SampleData")


class ClassificationDataset(Generic[T_SampleData, T_ClassId], Dataset[Tuple[T_SampleData, T_ClassId]], ABC):

    def __init__(self):
        super(Dataset, self).__init__()
        self.__description__: Optional[ClassificationDatasetDescription[T_ClassId]] = None
        self.__mapping__: Optional[ClassificationMapping[T_ClassId]] = None

    @abstractmethod
    def __getitem__(self, index) -> T_co:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def get_mapping(self) -> ClassificationMapping[T_ClassId]:
        if self.__mapping__ is None:
            self.__analyze__()

        return self.__mapping__

    def get_description(self) -> ClassificationDatasetDescription[T_ClassId]:
        if self.__description__ is None:
            self.__analyze__()

        return self.__description__

    def __str__(self) -> str:
        return f"size: {len(self)}\n" + str(self.get_description())

    def __analyze__(self):

        class_counts: Dict[T_ClassId, int] = dict()
        class_ids: Set[T_ClassId] = set()

        for i in range(0, len(self)):
            sample_data, class_id = self.__getitem__(i)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            class_ids.add(class_id)

        self.__description__ = ClassificationDatasetDescription(class_counts)
        self.__mapping__ = ClassificationMapping(sorted(class_ids))
