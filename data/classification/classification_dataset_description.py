from typing import Dict, TypeVar, Generic, Tuple, List, Any

import torch
from prettytable import PrettyTable

T_ClassId = TypeVar("T_ClassId")


class ClassificationDatasetDescription(Generic[T_ClassId]):
    def __init__(self, class_counts: Dict[T_ClassId, int]):
        super(ClassificationDatasetDescription, self).__init__()

        self.class_counts = class_counts
        self.num_classes = len(class_counts)
        self.class_weights: Dict[T_ClassId, float] = dict()
        self.class_ratios: Dict[T_ClassId, float] = dict()

        max_count = max(class_counts.values())
        total_count = sum(class_counts.values())

        for class_id, class_count in class_counts.items():
            self.class_weights[class_id] = 1 / (float(class_count) / float(total_count))
            self.class_ratios[class_id] = float(class_count) / float(max_count)

    def get_class_weights(self,
                          class_indices: List[int],
                          class_indices_to_ids: Dict[int, T_ClassId]) -> Dict[T_ClassId, float]:
        weights = dict()

        for class_index in class_indices:
            class_id = class_indices_to_ids[class_index]
            weight = self.class_weights[class_id]
            weights[class_id] = weight

        return weights

    def get_class_weight_tensor(self,
                                class_indices: List[int],
                                class_indices_to_ids: Dict[int, T_ClassId]) -> torch.Tensor:
        weights = list()

        for class_index in class_indices:
            class_id = class_indices_to_ids[class_index]
            weight = self.class_weights[class_id]
            weights.append(weight)

        return torch.tensor(weights)

    def to_table(self) -> PrettyTable:
        result = PrettyTable(field_names=["class_id", "count", "ratio", "weight"])

        for class_id in self.class_counts.keys():
            result.add_row([str(class_id), str(self.class_counts[class_id]), str(self.class_ratios[class_id]),
                            str(self.class_weights[class_id])])

        return result

    def __str__(self) -> str:
        return f"num_classes: {self.num_classes}\n" + self.to_table().get_string()
