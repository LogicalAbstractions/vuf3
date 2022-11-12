import json

from data.classification.video.video_classification_dataset_factories import create_video_classification_datasets, \
    create_video_classification_datamodule
from models.classification.classification_model_factories import create_classification_module
from models.classification.video.video_classification_model_configuration import VideoClassificationModelConfiguration
from models.classification.video.video_classification_model_description import VideoClassificationModelDescription
from models.classification.video.video_classification_model_factories import register_all_video_classification_models
from training.classification.classification_training_factories import run_classification_training
from utilities.configuration.configuration_reader import ConfigurationReader
from utilities.json import write_json_file
from utilities.random import initialize_random_numbers


def entrypoint():
    initialize_random_numbers()
    register_all_video_classification_models()

    configuration_reader = ConfigurationReader.create_from_cmdline(task="video_classification_training")
    datasets, weights, mapping = create_video_classification_datasets(configuration_reader)
    datamodule = create_video_classification_datamodule(configuration_reader, datasets, mapping)
    model = create_classification_module(configuration_reader, VideoClassificationModelConfiguration,
                                         mapping.num_classes, weights)

    metrics, confusion_matrix = run_classification_training(configuration_reader, VideoClassificationModelConfiguration,
                                                            datamodule,
                                                            model, mapping)

    description = VideoClassificationModelDescription.create(configuration_reader, mapping, metrics, confusion_matrix)

    write_json_file(configuration_reader.get_artifact_path() / "model.json", description)


if __name__ == "__main__":
    entrypoint()
