import enum


class ModelTask(enum.Enum):
    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"
    INFERENCE = "inf"
