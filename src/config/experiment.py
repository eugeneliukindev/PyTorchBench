import argparse
import logging
from pathlib import Path
from typing import Any, Literal

from src.utils import configure_logger
from .additional_schemas import ExperimentPaths
from .base_schemas import Config, LoggerConfig
from ..utils.general import load_yaml, save_yaml

log = logging.getLogger(__name__)


CRITICAL_COMPONENTS: dict[
    Literal["train", "test"],
    set[Literal["model", "dataset", "criterion", "optimizer", "scheduler"]],
] = {
    "train": {"model", "dataset", "criterion", "optimizer", "scheduler"},
    "test": {"model", "dataset", "criterion"},
}


def _check_critical_component_changes(
    current_config: dict[str, Any],
    previous_config: dict[str, Any],
    critical_components: set[Literal["model", "dataset", "criterion", "optimizer", "scheduler"]],
) -> None:
    def _compare_values(value1: Any, value2: Any, path: str) -> list[str]:
        differences = []

        if isinstance(value1, dict) and isinstance(value2, dict):
            all_keys = set(value1.keys()) | set(value2.keys())
            for key in all_keys:
                nested_path = f"{path}.{key}" if path else key
                differences.extend(_compare_values(value1.get(key), value2.get(key), nested_path))
        elif value1 != value2:
            differences.append(f"{path}: {value2} -> {value1}")

        return differences

    detected_changes = []

    for component in critical_components:
        current_value = current_config.get(component)
        previous_value = previous_config.get(component)
        detected_changes.extend(_compare_values(current_value, previous_value, component))

    if detected_changes:
        changes_summary = "\n".join(detected_changes)
        raise ValueError(
            f"Detected changes in critical experimental parameters:\n"
            f"{changes_summary}\n"
            f"Please create a new experiment with a trimmed root_path or revert the parameters."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def setup_experiment() -> Config:
    config_path = _parse_args().config
    config_dict = load_yaml(config_path)
    paths = ExperimentPaths.model_validate(config_dict.get("experiment", {}).get("paths", {}))
    logger_config = LoggerConfig.model_validate(config_dict.get("logger", {}))

    for field in paths.model_fields_set:
        path = getattr(paths, field)  # type: Path
        if field == "root_path":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    configure_logger(level=logger_config.level, filename=paths.logs_path)

    config: Config = Config.model_validate(config_dict)
    if config.train.enabled:
        if config.optimizer is None:
            raise ValueError("Optimizer is not configured, but training is enabled.")
        if config.scheduler is None:
            raise ValueError("Scheduler is not configured, but training is enabled.")

    if paths.config_snapshot.exists():
        config_snapshot = load_yaml(paths.config_snapshot)
        critical_components = CRITICAL_COMPONENTS["train"] if config.train.enabled else CRITICAL_COMPONENTS["test"]
        _check_critical_component_changes(
            current_config=config_dict,
            previous_config=config_snapshot,
            critical_components=critical_components,
        )
    else:
        save_yaml(paths.config_snapshot, config_dict)

    log.info("Successfully loaded experiment config: %s", config_path)
    return config
