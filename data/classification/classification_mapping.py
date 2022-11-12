from typing import Generic, List, Dict

from prettytable import PrettyTable

from data.classification.classification_dataset_description import T_ClassId


class ClassificationMappingError(Exception):
    pass


class ClassificationMapping(Generic[T_ClassId]):
    def __init__(self, class_ids: List[T_ClassId]):
        super(ClassificationMapping, self).__init__()

        self.class_indices = [i for i, _ in enumerate(class_ids)]
        self.class_ids = class_ids

        self.class_ids_to_index: Dict[T_ClassId, int] = {
            c: i for i, c in enumerate(class_ids)
        }
        self.class_indices_to_class_id: Dict[int, T_ClassId] = {
            i: c for i, c in enumerate(class_ids)
        }

        self.num_classes = len(class_ids)

    def get_class_index(self, class_id: T_ClassId) -> int:
        if class_id in self.class_ids_to_index:
            return self.class_ids_to_index[class_id]

        raise ClassificationMappingError(f"Could not find class id {class_id} in mapping table")

    def get_class_id(self, class_index: int) -> T_ClassId:
        if class_index in self.class_indices_to_class_id:
            return self.class_indices_to_class_id[class_index]

        raise ClassificationMappingError(f"Could not find class index {class_index} in mapping table")

    def to_table(self) -> PrettyTable:
        result = PrettyTable(field_names=["index", "id"])

        for class_index in self.class_indices:
            class_id = self.class_indices_to_class_id[class_index]
            result.add_row([str(class_index), str(class_id)])

        return result

    def __str__(self) -> str:
        return self.to_table().get_string()
