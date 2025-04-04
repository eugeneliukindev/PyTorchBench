import csv
import importlib
import logging
from os import PathLike
from typing import Any, Literal, overload
from typing import Mapping, Sequence

import yaml
from matplotlib import pyplot as plt

from src.training import TrainMetrics, ValidationMetrics, EpochMetrics

log = logging.getLogger(__name__)


def get_object_from_import_path(import_path: str) -> Any:
    if not isinstance(import_path, str):
        raise TypeError(f"Expected str, got {type(import_path).__name__}")
    parts = import_path.split(".")
    module = importlib.import_module(parts[0])
    result = module
    for part in parts[1:]:
        result = getattr(result, part)
    log.info(
        "Successfully imported object: %s from %s",
        getattr(result, "__name__", type(result).__name__),
        import_path,
    )
    return result


def save_yaml(path: str | PathLike[str], data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        log.info("Saved YAML file: %s", path)
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
        )


def load_yaml(path: str | PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        log.info("Loaded YAML file: %s", path)
        return yaml.safe_load(f)


def save_to_csv(
    path: str | PathLike[str],
    data: Mapping[str, Any] | Sequence[Sequence[Any]],
    mode: Literal["w", "a", "x"] = "a",
) -> None:
    if mode not in ("w", "a", "x"):
        raise ValueError(f"Invalid mode: {mode}. Expected one of 'w', 'a', 'x'.")
    with open(path, mode, newline="") as f:
        if isinstance(data, Mapping):
            dict_writer = csv.DictWriter(f, fieldnames=list(data))
            if f.tell() == 0:
                dict_writer.writeheader()
            dict_writer.writerow(data)
        elif (
            isinstance(data, Sequence)
            and not isinstance(data, (str, bytes))
            and all(isinstance(row, Sequence) for row in data if not isinstance(row, (str, bytes)))
        ):
            seq_writer = csv.writer(f)
            if f.tell() == 0:
                seq_writer.writerow(data[0])
                seq_writer.writerows(data[1:])
            else:
                seq_writer.writerows(data)
        else:
            raise TypeError("Unsupported data type. Expected Mapping[str, Any] or Sequence[Sequence[Any]].")
        log.info("Saved data to CSV file: %s", path)


@overload
def load_from_csv(path: str | PathLike[str], reader_type: Literal["dict"] = "dict") -> list[dict[str, str]]: ...


@overload
def load_from_csv(path: str | PathLike[str], reader_type: Literal["list"]) -> list[list[str]]: ...


def load_from_csv(
    path: str | PathLike[str],
    reader_type: Literal["dict", "list"] = "dict",
) -> list[dict[str, str]] | list[list[str]]:
    """Loads a CSV file and returns either a list of dictionaries or a list of lists.

    Args:
        path (str | PathLike[str]): Path to the CSV file.
        reader_type (Literal["dict", "list"], optional): Type of data to return.
            "dict" for a list of dictionaries, "list" for a list of lists.
            Defaults to "dict".

    Returns:
        list[dict[str, str]] | list[list[str]]: The CSV data as a list.

    Raises:
        ValueError: If reader_type is invalid.
    """
    if reader_type not in ("dict", "list"):
        raise ValueError(f"Invalid reader type: {reader_type!r}. Expected 'dict' or 'list'")
    log.info("Loading CSV file: %s", path)
    with open(path, newline="", encoding="utf-8") as f:
        if reader_type == "dict":
            return list(csv.DictReader(f))
        return list(csv.reader(f))


def _extract_metrics(metrics_history: Sequence[EpochMetrics]) -> list[tuple[int, float, float, float]]:
    return list(zip(*[(m.loss, m.accuracy, m.time, m.epoch) for m in metrics_history]))


def _validate_save_graph(
    path: str | PathLike[str],
    train_metrics_history: list[TrainMetrics],
    val_metrics_history: list[ValidationMetrics],
) -> None:
    if not isinstance(path, (str, PathLike)):
        raise TypeError(f"Expected str or PathLike, got {type(path).__name__}")
    if not isinstance(train_metrics_history, list):
        raise TypeError(f"Expected list, got {type(train_metrics_history).__name__}")
    if not isinstance(val_metrics_history, list):
        raise TypeError(f"Expected list, got {type(val_metrics_history).__name__}")

    if not train_metrics_history or not val_metrics_history:
        raise ValueError("Both train_metrics_history and val_metrics_history must be non-empty lists")
    if not all(isinstance(m, (TrainMetrics, ValidationMetrics)) for m in train_metrics_history + val_metrics_history):
        raise ValueError(
            "train_metrics_history and val_metrics_history must contain only TrainMetrics and ValidationMetrics"
        )


def save_graph(
    path: str | PathLike[str],
    train_metrics_history: list[TrainMetrics],
    val_metrics_history: list[ValidationMetrics],
) -> None:
    _validate_save_graph(
        path=path,
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
    )
    train_losses, train_accuracy, _, train_epochs = _extract_metrics(train_metrics_history)
    val_losses, val_accuracy, _, val_epochs = _extract_metrics(val_metrics_history)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(train_epochs, train_losses, label="Train Loss", color="blue", marker="o")
    ax1.plot(val_epochs, val_losses, label="Val Loss", color="orange", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train and Validation Loss")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", fontsize=10)

    best_train = min(train_metrics_history, key=lambda m: m.loss)
    ax1.annotate(
        f"Min: {best_train.loss:.4f}",
        xy=(best_train.epoch, best_train.loss),
        xytext=(5, 5),
        textcoords="offset points",
        color="blue",
    )
    best_val = min(val_metrics_history, key=lambda m: m.loss)
    ax1.annotate(
        f"Min: {best_val.loss:.4f}",
        xy=(best_val.epoch, best_val.loss),
        xytext=(5, 5),
        textcoords="offset points",
        color="orange",
    )

    ax2 = ax1.twinx()
    ax2.plot(train_epochs, train_accuracy, label="Train Accuracy", color="green", linestyle="--")
    ax2.plot(val_epochs, val_accuracy, label="Val Accuracy", color="red", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        log.info("Saved loss graph to %s", path)
    except Exception as e:
        log.exception("Failed to save graph to %s: %s", path, str(e))
        raise OSError(f"Could not save graph to {path}: {str(e)}")
    finally:
        plt.close(fig)
