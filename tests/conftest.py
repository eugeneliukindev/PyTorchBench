from typing import Any, Final

import pytest
import torchvision

from src.training import ValidationMetrics, TrainMetrics

NO_RESULT: Final = object()


@pytest.fixture
def config_dict(tmp_path) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "fruits_classification",
            "paths": {
                "root_path": str(tmp_path),
                "model_path": "model.pth",
                "best_model_path": "best_model.pth",
                "optimizer_path": "optimizer.pth",
                "best_optimizer_path": "best_optimizer.pth",
                "train_metrics_path": "train_metrics.csv",
                "val_metrics_path": "val_metrics.csv",
                "test_metrics_path": "test_metrics.csv",
                "graph_path": "graph.png",
                "config_snapshot": "details.yaml",
                "logs_path": "logs.log",
            },
        },
        "logger": {"level": "DEBUG"},
        "train": {"enabled": True, "epochs": 50},
        "test": {"enabled": True},
        "dataset": {
            "obj": "src.data.datasets.ImageClassificationDataset",
            "init_params": {
                "path": str(tmp_path / "dataset"),
                "transforms": {
                    "obj": "torchvision.transforms.v2.Compose",
                    "init_params": {
                        "transforms": [
                            {"obj": "torchvision.transforms.v2.RGB", "init_params": {}},
                            {
                                "obj": "torchvision.transforms.v2.Resize",
                                "init_params": {"size": [224, 224]},
                            },
                            {
                                "obj": "torchvision.transforms.v2.ToImage",
                                "init_params": {},
                            },
                            {
                                "obj": "torchvision.transforms.v2.ToDtype",
                                "init_params": {
                                    "dtype": {
                                        "obj": "torch.float32",
                                        "init_params": None,
                                    },
                                    "scale": True,
                                },
                            },
                            {
                                "obj": "torch.nn.Sequential",
                                "init_params": {
                                    "args": [
                                        {
                                            "obj": "torch.nn.Conv2d",
                                            "init_params": {
                                                "in_channels": 3,
                                                "out_channels": 16,
                                                "kernel_size": 3,
                                            },
                                        },
                                        {
                                            "obj": "torch.nn.ReLU",
                                            "init_params": {"inplace": True},
                                        },
                                    ]
                                },
                            },
                        ]
                    },
                },
            },
            "post_params": {"batch_size": 64},
        },
        "model": {
            "obj": "torchvision.models.resnet101",
            "init_params": {
                "weights": {
                    "obj": "torchvision.models.ResNet101_Weights.DEFAULT",
                    "init_params": None,
                },
            },
            "post_params": {"out_features": 12, "freeze_pretrained_weights": True},
        },
        "criterion": {
            "obj": "torch.nn.CrossEntropyLoss",
            "init_params": {},
        },
        "optimizer": {
            "obj": "torch.optim.Adam",
            "init_params": {},
        },
        "scheduler": {
            "obj": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_params": {},
        },
        "metric_tracker": {
            "obj": "torchmetrics.Accuracy",
            "init_params": {
                "task": "multiclass",
                "out_features": 12,
            },
        },
    }


@pytest.fixture(scope="function")
def config_yaml(tmp_path) -> str:
    return (
        "experiment:\n"
        "  name: fruits_classification\n"
        "  paths:\n"
        f"    root_path: {str(tmp_path)}\n"
        "    model_path: model.pth\n"
        "    best_model_path: best_model.pth\n"
        "    optimizer_path: optimizer.pth\n"
        "    best_optimizer_path: best_optimizer.pth\n"
        "    train_metrics_path: train_metrics.csv\n"
        "    val_metrics_path: val_metrics.csv\n"
        "    test_metrics_path: test_metrics.csv\n"
        "    graph_path: graph.png\n"
        "    config_snapshot: details.yaml\n"
        "    logs_path: logs.log\n"
        "logger:\n"
        "  level: DEBUG\n"
        "train:\n"
        "  enabled: true\n"
        "  epochs: 50\n"
        "test:\n"
        "  enabled: true\n"
        "dataset:\n"
        "  obj: src.data.datasets.ImageClassificationDataset\n"
        "  init_params:\n"
        f"    path: {str(tmp_path / 'dataset')}\n"
        "    transforms:\n"
        "      obj: torchvision.transforms.v2.Compose\n"
        "      init_params:\n"
        "        transforms:\n"
        "        - obj: torchvision.transforms.v2.RGB\n"
        "          init_params: {}\n"
        "        - obj: torchvision.transforms.v2.Resize\n"
        "          init_params:\n"
        "            size:\n"
        "            - 224\n"
        "            - 224\n"
        "        - obj: torchvision.transforms.v2.ToImage\n"
        "          init_params: {}\n"
        "        - obj: torchvision.transforms.v2.ToDtype\n"
        "          init_params:\n"
        "            dtype:\n"
        "              obj: torch.float32\n"
        "              init_params: null\n"
        "            scale: true\n"
        "        - obj: torch.nn.Sequential\n"
        "          init_params:\n"
        "            args:\n"
        "            - obj: torch.nn.Conv2d\n"
        "              init_params:\n"
        "                in_channels: 3\n"
        "                out_channels: 16\n"
        "                kernel_size: 3\n"
        "            - obj: torch.nn.ReLU\n"
        "              init_params:\n"
        "                inplace: true\n"
        "  post_params:\n"
        "    batch_size: 64\n"
        "model:\n"
        "  obj: torchvision.models.resnet101\n"
        "  init_params:\n"
        "    weights:\n"
        "      obj: torchvision.models.ResNet101_Weights.DEFAULT\n"
        "      init_params: null\n"
        "  post_params:\n"
        "    out_features: 12\n"
        "    freeze_pretrained_weights: true\n"
        "criterion:\n"
        "  obj: torch.nn.CrossEntropyLoss\n"
        "  init_params: {}\n"
        "optimizer:\n"
        "  obj: torch.optim.Adam\n"
        "  init_params: {}\n"
        "scheduler:\n"
        "  obj: torch.optim.lr_scheduler.ReduceLROnPlateau\n"
        "  init_params: {}\n"
        "metric_tracker:\n"
        "  obj: torchmetrics.Accuracy\n"
        "  init_params:\n"
        "    task: multiclass\n"
        "    out_features: 12\n"
    )


@pytest.fixture(scope="session")
def train_metrics_history() -> list[TrainMetrics]:
    return [
        TrainMetrics(loss=1.0, accuracy=0.5, time=120.0, epoch=1),
        TrainMetrics(loss=0.7, accuracy=0.65, time=120.0, epoch=2),
        TrainMetrics(loss=0.5, accuracy=0.75, time=115.5, epoch=3),
        TrainMetrics(loss=0.35, accuracy=0.82, time=118.0, epoch=4),
    ]


@pytest.fixture(scope="session")
def val_metrics_history() -> list[ValidationMetrics]:
    return [
        ValidationMetrics(loss=0.7, accuracy=0.65, time=120.0, epoch=1),
        ValidationMetrics(loss=0.5, accuracy=0.8, time=145.0, epoch=2),
    ]


@pytest.fixture(scope="session")
def resnet18_model():
    return torchvision.models.resnet18()


@pytest.fixture(scope="session")
def vgg16_model():
    return torchvision.models.vgg16()
