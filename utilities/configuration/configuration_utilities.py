from typing import Optional


def is_str_configuration_value_valid(value: Optional[str]):
    return value is not None and len(value) > 0
