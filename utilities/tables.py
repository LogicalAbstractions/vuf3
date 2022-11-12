from typing import Dict, Any, List, Tuple

from numpy import ndarray
from prettytable import PrettyTable
from torch import Tensor

from data.classification.classification_mapping import ClassificationMapping


def list_to_table(values: List[Tuple[str, Any]]) -> str:
    table = PrettyTable()

    for key, value in values:
        table.add_row([key, str(value)])

    return table.get_string()


def add_sub_table(parent: PrettyTable, sub_table: PrettyTable) -> PrettyTable:
    parent.add_row(["", sub_table.get_string()])
    return parent


def confusion_matrix_to_table(mapping: ClassificationMapping[str], matrix: Tensor) -> PrettyTable:
    matrix_array: ndarray = matrix.cpu().detach().numpy()
    result = PrettyTable(field_names=["id"] + mapping.class_ids)

    for y in range(0, mapping.num_classes):
        row_values = [mapping.class_ids[y]]
        for x in range(0, mapping.num_classes):
            row_values.append(str(matrix_array[x, y]))

        result.add_row(row_values)

    return result


def table_to_json_data(table: PrettyTable, include_header: bool= False) -> List:
    result = list()

    if include_header:
        result.append(table.field_names)

    for row in table.rows:
        result.append(dict(zip(table.field_names, row)))

    return result
