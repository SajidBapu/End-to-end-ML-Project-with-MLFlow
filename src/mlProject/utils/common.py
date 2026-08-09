import json
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.
    """
    with path_to_yaml.open("r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)

    if content is None:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")

    return ConfigBox(content)


def create_directories(paths: list[Path], verbose: bool = True) -> None:
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Created directory: {path}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Save dictionary data to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path: Path) -> ConfigBox:
    """
    Load JSON data from a file.
    """
    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    return ConfigBox(content)


def save_bin(data: Any, path: Path) -> None:
    """
    Save a Python object using joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def load_bin(path: Path) -> Any:
    """
    Load a Python object saved using joblib.
    """
    return joblib.load(path)