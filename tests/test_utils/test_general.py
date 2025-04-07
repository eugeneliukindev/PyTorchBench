import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
import torchvision
from PIL import Image
from _pytest.fixtures import SubRequest
from torch import nn as nn

from src.training import TrainMetrics, ValidationMetrics
from src.utils import (
    get_object_from_import_path,
    save_yaml,
    load_yaml,
    save_to_csv,
    load_from_csv,
    save_graph,
)
from tests.conftest import NO_RESULT, config_dict


@pytest.fixture
def train_metrics_history() -> list[TrainMetrics]:
    return [
        TrainMetrics(loss=1.0, accuracy=0.5, time=120.0, epoch=1),
        TrainMetrics(loss=0.7, accuracy=0.65, time=120.0, epoch=2),
    ]


@pytest.fixture
def val_metrics_history() -> list[ValidationMetrics]:
    return [
        ValidationMetrics(loss=0.7, accuracy=0.65, time=120.0, epoch=1),
        ValidationMetrics(loss=0.5, accuracy=0.8, time=145.0, epoch=2),
    ]


@pytest.fixture
def temp_file(tmp_path_factory):
    def _create_file(filename: str) -> Path:
        tmp_dir = tmp_path_factory.mktemp("test_files")
        return tmp_dir / filename

    return _create_file


@pytest.fixture
def yaml_file(temp_file) -> Path:
    return temp_file("test.yaml")


@pytest.mark.parametrize(
    "import_path, expected_result, expectation",
    [
        ("torchvision.models.resnet18", torchvision.models.resnet18, does_not_raise()),
        ("torch.nn.Linear", nn.Linear, does_not_raise()),
        (
            "invalid.path",
            NO_RESULT,
            pytest.raises(ModuleNotFoundError, match="No module named 'invalid'"),
        ),
        (42, NO_RESULT, pytest.raises(TypeError, match="Expected str, got int")),
        (
            "path_without_dot",
            NO_RESULT,
            pytest.raises(ImportError, match="No module named 'path_without_dot'"),
        ),
    ],
    ids=[
        # success cases
        "resnet18",
        "linear_layer",
        # exception cases
        "invalid_module_name",
        "invalid_import_path",
        "invalid_type",
    ],
)
def test_get_object_from_import_path(import_path, expected_result, expectation):
    with expectation:
        result = get_object_from_import_path(import_path)
        assert result == expected_result


class TestYaml:
    def test_save_yaml(self, yaml_file, config_dict, config_yaml):
        save_yaml(path=yaml_file, data=config_dict)
        result = yaml_file.read_text()
        assert result == config_yaml

    def test_load_yaml(self, yaml_file, config_dict):
        save_yaml(path=yaml_file, data=config_dict)
        result = load_yaml(path=yaml_file)
        assert result == config_dict


class TestCSV:
    @pytest.mark.parametrize(
        "data, mode, expected_result, expectation",
        [
            (
                {"epoch": 1, "loss": 1.0, "accuracy": 0.5},
                "w",
                "epoch,loss,accuracy\n1,1.0,0.5\n",
                does_not_raise(),
            ),
            (
                [["epoch", "loss", "accuracy"], [1, 1.0, 0.5], [2, 0.7, 0.65]],
                "w",
                "epoch,loss,accuracy\n1,1.0,0.5\n2,0.7,0.65\n",
                does_not_raise(),
            ),
            (
                {"epoch": 1, "loss": 1.0, "accuracy": 0.5},
                "a",
                "epoch,loss,accuracy\n1,1.0,0.5\n",
                does_not_raise(),
            ),
            (
                [["epoch", "loss"], [1, 1.0]],
                "x",
                "epoch,loss\n1,1.0\n",
                does_not_raise(),
            ),
            (
                [["epoch", "loss"], [1]],
                "w",
                "epoch,loss\n1\n",
                does_not_raise(),
            ),
            (
                42,
                "w",
                NO_RESULT,
                pytest.raises(
                    TypeError,
                    match=re.escape(
                        "Unsupported data type. Expected Mapping[str, Any] or Sequence[Sequence[Any]].",
                    ),
                ),
            ),
            (
                {"epoch": 1, "loss": 1.0},
                "z",
                NO_RESULT,
                pytest.raises(ValueError, match="Invalid mode: z. Expected one of 'w', 'a', 'x'."),
            ),
        ],
        ids=[
            # success cases
            "dict_write",
            "list_write",
            "dict_append",
            "list_exclusive",
            "incomplete_list",
            # exception cases
            "invalid_type",
            "invalid_mode",
        ],
    )
    def test_save_to_csv(self, tmp_path, data, mode, expected_result, expectation):
        path = tmp_path / "test.csv"
        with expectation:
            save_to_csv(path=path, data=data, mode=mode)
        if expected_result is not NO_RESULT:
            result = path.read_text()
            assert result == expected_result

    @pytest.mark.parametrize(
        "data, reader_type, expected_result, expectation",
        [
            (
                "epoch,loss,accuracy\n1,1.0,0.5\n",
                "list",
                [["epoch", "loss", "accuracy"], ["1", "1.0", "0.5"]],
                does_not_raise(),
            ),
            (
                "epoch,loss,accuracy\n1,1.0,0.5\n2,0.7,0.65\n",
                "dict",
                [
                    {"epoch": "1", "loss": "1.0", "accuracy": "0.5"},
                    {"epoch": "2", "loss": "0.7", "accuracy": "0.65"},
                ],
                does_not_raise(),
            ),
            (
                "Something",
                "string",
                NO_RESULT,
                pytest.raises(
                    ValueError,
                    match="Invalid reader type: 'string'. Expected 'dict' or 'list'",
                ),
            ),
        ],
        ids=[
            # success cases
            "list_reader",
            "dict_reader",
            # exception cases
            "invalid_reader_type",
        ],
    )
    def test_load_from_csv(self, tmp_path, data, reader_type, expected_result, expectation):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(data)
        with expectation:
            result = load_from_csv(csv_path, reader_type)
            assert result == expected_result


@pytest.mark.parametrize(
    "train_metrics, val_metrics, expectation",
    [
        (
            "train_metrics_history",
            "val_metrics_history",
            does_not_raise(),
        ),
        (
            [0, 1, 2, 4],
            "val_metrics_history",
            pytest.raises(
                ValueError,
                match="train_metrics_history and val_metrics_history must contain only TrainMetrics and ValidationMetrics",
            ),
        ),
        (
            [0, 1, 2, 4],
            [0, 1, 2, 4],
            pytest.raises(
                ValueError,
                match="train_metrics_history and val_metrics_history must contain only TrainMetrics and ValidationMetrics",
            ),
        ),
        (
            "train_metrics_history",
            [0, 1, 2, 4],
            pytest.raises(
                ValueError,
                match="train_metrics_history and val_metrics_history must contain only TrainMetrics and ValidationMetrics",
            ),
        ),
        (
            "train_metrics_history",
            "val_metrics_history",
            pytest.raises(TypeError, match="Expected str or PathLike, got int"),
        ),
        (
            [],
            [],
            pytest.raises(
                ValueError,
                match="Both train_metrics_history and val_metrics_history must be non-empty lists",
            ),
        ),
    ],
    ids=[
        # success cases
        "valid_metrics_and_extraction",
        # exception cases
        "invalid_train_metrics_for_extraction",
        "invalid_metrics_for_save",
        "invalid_val_metrics_for_save",
        "invalid_path",
        "empty_metrics",
    ],
)
def test_save_graph(request: SubRequest, tmp_path, train_metrics, val_metrics, expectation):
    if isinstance(train_metrics, str):
        train_metrics = request.getfixturevalue(train_metrics)
    if isinstance(val_metrics, str):
        val_metrics = request.getfixturevalue(val_metrics)
    path = tmp_path / "test.png" if request.node.callspec.id != "invalid_path" else 42
    with expectation:
        save_graph(
            path=path,
            train_metrics_history=train_metrics,
            val_metrics_history=val_metrics,
        )
        with Image.open(path) as img:
            img.verify()
