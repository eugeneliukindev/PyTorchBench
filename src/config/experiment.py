from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, Literal


from src.utils import configure_logger
from .additional_schemas import ExperimentPaths
from .base_schemas import Config, LoggerConfig
from ..utils.general import load_yaml, save_yaml

import logging

log = logging.getLogger(__name__)


class CriticalComponent(StrEnum):
    MODEL = "model"
    DATASET = "dataset"
    CRITERION = "criterion"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"


CRITICAL_COMPONENTS: dict[Literal["train", "test"], set[CriticalComponent]] = {
    "train": {
        CriticalComponent.MODEL,
        CriticalComponent.DATASET,
        CriticalComponent.CRITERION,
        CriticalComponent.OPTIMIZER,
        CriticalComponent.SCHEDULER,
    },
    "test": {
        CriticalComponent.MODEL,
        CriticalComponent.DATASET,
        CriticalComponent.CRITERION,
    },
}


def _check_critical_component_changes(
    current_config: dict[str, Any],
    previous_config: dict[str, Any],
    critical_components: set[CriticalComponent],
) -> None:
    def _compare_values(value1: Any, value2: Any, path: str) -> list[str]:
        differences = []

        if isinstance(value1, dict) and isinstance(value2, dict):
            all_keys = set(value1.keys()) | set(value2.keys())
            for key in all_keys:
                nested_path = f"{path}.{key}" if path else key
                differences.extend(
                    _compare_values(value1.get(key), value2.get(key), nested_path)
                )
        elif value1 != value2:
            differences.append(f"{path}: {value2} -> {value1}")

        return differences

    detected_changes = []

    for component in critical_components:
        component_name = component.value
        current_value = current_config.get(component_name)
        previous_value = previous_config.get(component_name)
        detected_changes.extend(
            _compare_values(current_value, previous_value, component_name)
        )

    if detected_changes:
        changes_summary = "\n".join(detected_changes)
        raise ValueError(
            f"Detected changes in critical experimental parameters:\n"
            f"{changes_summary}\n"
            f"Please create a new experiment with a trimmed root_path or revert the parameters."
        )


def setup_experiment(config_path: str | PathLike[str]) -> Config:
    config_dict = load_yaml(config_path)
    paths = ExperimentPaths.model_validate(
        config_dict.get("experiment", {}).get("paths", {})
    )
    logger_config = LoggerConfig.model_validate(config_dict.get("logger", {}))

    for field in paths.model_fields_set:
        path = getattr(paths, field)  # type: Path
        path.parent.mkdir(parents=True, exist_ok=True)

    configure_logger(level=logger_config.level, filename=paths.logs_path)

    config: Config = Config.model_validate(config_dict)
    if config.train.enabled:
        if config.optimizer is None:
            raise ValueError("Optimizer is not configured, but training is enabled.")
        if config.scheduler is None:
            raise ValueError("Scheduler is not configured, but training is enabled.")
        save_yaml(paths.config_snapshot, config_dict)
    saved_details = load_yaml(paths.config_snapshot)
    _check_critical_component_changes(
        current_config=config_dict,
        previous_config=saved_details,
        critical_components=(
            CRITICAL_COMPONENTS["train"]
            if config.train.enabled
            else CRITICAL_COMPONENTS["test"]
        ),
    )
    log.info("Successfully loaded experiment config: %s", config_path)
    return config
