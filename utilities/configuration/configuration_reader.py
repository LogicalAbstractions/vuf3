import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Type, Any, Dict, Callable, TypeVar


def __merge_dict__(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict()

    for k, v in a.items():
        result[k] = a[k]

    for k, v in b.items():
        result[k] = b[k]

    return result


T_Configuration = TypeVar("T_Configuration")


class ConfigurationReader:
    def __init__(self,
                 environment: str,
                 search_paths: List[Path],
                 task: str,
                 experiment: Optional[str],
                 artifact_root_path: Path):
        self.environment = environment
        self.search_paths = search_paths
        self.task = task
        self.experiment = experiment
        self.artifact_root_path = artifact_root_path

        print(f"Environment: {environment}")
        print(f"Experiment: {experiment}")

        print("Search paths:")
        for search_path in search_paths:
            print(f"{search_path}, exists: {search_path.exists()}")

    def read_object(self, type: Type, id: Optional[str] = None):
        actual_id = id if id is not None else type.id
        json_data = self.read_file_raw(actual_id)

        return type.from_dict(json_data)

    def read_file_raw(self, id: str) -> Dict[str, Any]:
        filename = f"{id.lower()}.json"
        json_data: List[Dict[str, Any]] = list()

        for search_path in self.search_paths:
            file_path = search_path / filename
            self.__read_file_with_environment__(file_path, json_data)

        result: Dict[str, any] = dict()

        for entry in json_data:
            result = __merge_dict__(result, entry)

        return result

    def get_artifact_path(self, ensure_created: bool = False):
        base_path = self.artifact_root_path / self.task.lower()

        if self.experiment is not None:
            base_path = base_path / self.experiment.lower()

        final_path = base_path / self.environment

        if ensure_created and not os.path.exists(final_path):
            os.makedirs(final_path)

        return final_path

    @staticmethod
    def create_from_cmdline(configuration_root_path: Path = Path("./configuration"),
                            artifact_root_path: Path = Path("./artifacts"),
                            task: str = "video_classification_training",
                            arg_parser_customization: Optional[
                                Callable[[ArgumentParser], None]] = None) -> 'ConfigurationReader':

        argument_parser = ArgumentParser()
        argument_parser.add_argument('-e', '--environment', default="dev", choices=["dev", "test", "prod"])
        argument_parser.add_argument('-x', '--experiment', default="mvit_01")

        if arg_parser_customization is not None:
            arg_parser_customization(argument_parser)

        arguments = argument_parser.parse_args()

        search_paths = [configuration_root_path]
        task_path = configuration_root_path / "tasks" / task

        search_paths.append(task_path)

        if arguments.experiment is not None:
            experiment_path = task_path / "experiments" / arguments.experiment
            search_paths.append(experiment_path)

        return ConfigurationReader(arguments.environment, search_paths, task, arguments.experiment, artifact_root_path)

    def __read_file_with_environment__(self, path: Path, result: List[Dict[str, Any]]) -> None:

        if self.__read_single_file__(path, result):
            environment_path = path.with_suffix(f".{self.environment.lower()}.json")
            self.__read_single_file__(environment_path, result)

    def __read_single_file__(self, path: Path, result: List[Dict[str, Any]]) -> bool:
        if path.exists():
            with open(path, "r") as f:
                result.append(json.load(f))
                if self.environment == "dev":
                    print(f"Read configuration from: {path}")

                return True

        return False
