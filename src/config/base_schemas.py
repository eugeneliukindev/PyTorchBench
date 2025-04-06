from pathlib import Path
from typing import Any, Mapping, Literal, Annotated, Sequence, TypeVar, overload

import torch
from pydantic import BaseModel, ConfigDict, model_validator, Field

from src.config.additional_schemas import ExperimentPaths, ModelPostParams, DatasetPostParams
from src.utils import get_object_from_import_path

_T = TypeVar("_T")


class ObjectFactory(BaseModel):
    """
    A configuration class that recursively processes and resolves object specifications
    and their parameters into actual Python objects.

    This class supports importing objects from string paths, processing nested configurations,
    and handling callable objects with parameters. It is designed to work with complex nested
    structures including dictionaries, sequences, and string-based import paths.

    Args:
        obj (Any): The main object to process, either as a direct object or a string import path.
        init_params (Mapping[str, Any] | None): Optional parameters for the object, if it is callable.

    Examples:
        >>> test1 = ObjectFactory(
        ...     obj="torchvision.models.resnet101",
        ...     init_params={
        ...         "weights": {
        ...             "obj": "torchvision.models.ResNet101_Weights.DEFAULT",
        ...             "init_params": None
        ...         }
        ...     }
        ... )
        >>> test1  # doctest: +ELLIPSIS
        ObjectFactory(obj=<function resnet101 at ...>, init_params={'weights': ResNet101_Weights.IMAGENET1K_V2})
        >>> test1.create()  # doctest: +ELLIPSIS
        ResNet(...)

        >>> test2 = ObjectFactory(
        ...     obj=torch.nn.CrossEntropyLoss,
        ...     init_params={}
        ... )
        >>> test2
        ObjectFactory(obj=<class 'torch.nn.modules.loss.CrossEntropyLoss'>, init_params={})
        >>> test2.create()
        CrossEntropyLoss()

        >>> test3 = ObjectFactory(
        ...     obj="torch.nn.Sequential",
        ...     init_params={
        ...         "args": [
        ...             {"obj": "torch.nn.Conv2d", "init_params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}},
        ...             {"obj": "torch.nn.ReLU", "init_params": {"inplace": True}}
        ...         ]
        ...     }
        ... )
        >>> test3  # doctest: +ELLIPSIS
        ObjectFactory(obj=<class 'torch.nn.modules.container.Sequential'>, init_params={'args': [...]})
        >>> test3.create()  # doctest: +ELLIPSIS
        Sequential(...)

    Notes:
        If `init_params` is `None`, the `create` method returns the object passed to `obj` as is.
        If `init_params` is provided as a `Mapping`, the `create` method calls `obj(*args, **kwargs)` where
        `args` is taken from `init_params["args"]` if present, and `kwargs` includes remaining parameters
        combined with any additional parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obj: Any
    init_params: Mapping[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "obj" not in values:
            raise ValueError("Obj must be provided")

        values = values.copy()
        obj = values["obj"]
        obj = get_object_from_import_path(obj) if isinstance(obj, str) else obj
        init_params = values.get("init_params")

        if callable(obj):
            if isinstance(init_params, Mapping):
                init_params = cls._process_params(init_params)
            elif init_params is None:
                init_params = None
            else:
                raise ValueError("init_params must be a Mapping when obj is callable")
        else:
            if init_params is not None:
                raise ValueError("init_params must be None when obj is not callable")

        values["obj"] = obj
        values["init_params"] = init_params
        return values

    @classmethod
    def _process_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        processed = {}
        for key, value in params.items():
            if key == "path":
                processed[key] = Path(value)
            else:
                processed[key] = cls._process_value(value)
        return processed

    @classmethod
    def _process_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            if "obj" in value:
                return ObjectFactory(**value).create()
            else:
                return cls._process_params(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls._process_value(item) for item in value]
        return value

    @overload
    def create(self, expected_obj: None = None, **additional_params: Any) -> Any: ...

    @overload
    def create(self, expected_obj: type[_T], **additional_params: Any) -> _T: ...

    def create(
        self,
        expected_obj: type[_T] | None = None,
        **additional_params: Any,
    ) -> Any | _T:
        if callable(self.obj) and self.init_params is not None:
            init_params = dict(self.init_params) | additional_params
            if "args" in init_params:
                args = init_params.pop("args")
                instance = self.obj(*args, **init_params)
            else:
                instance = self.obj(**init_params)
        else:
            instance = self.obj
        if expected_obj is not None and not isinstance(instance, expected_obj):
            raise ValueError(f"Expected object of type {expected_obj} but got {type(instance)}")
        return instance


class ExperimentConfig(BaseModel):
    name: str
    paths: ExperimentPaths


class LoggerConfig(BaseModel):
    level: Literal["INFO", "DEBUG", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "NOTSET"]


class TrainConfig(BaseModel):
    enabled: bool
    epochs: Annotated[int, Field(ge=1)]


class ModelConfig(ObjectFactory):
    post_params: ModelPostParams


class CriterionConfig(ObjectFactory):
    pass


class OptimizerConfig(ObjectFactory):
    pass


class SchedulerConfig(ObjectFactory):
    pass


class DatasetConfig(ObjectFactory):
    post_params: DatasetPostParams


class MetricTrackerConfig(ObjectFactory):
    pass


class TestConfig(BaseModel):
    enabled: bool


class Config(BaseModel):
    """
    A configuration class that defines the structure for an experiment setup, including model, dataset,
    training, testing, and logging configurations. Inherits from `BaseModel` (assumed to be from a library
    like Pydantic for validation).

    This class encapsulates all necessary settings to run a machine learning experiment, supporting nested
    object specifications resolved via an `ObjectFactory`-like mechanism (assumed for `obj` and `init_params`).

    Args:
        experiment (ExperimentConfig): Configuration for experiment paths and metadata.
        model (ModelConfig): Configuration for the model architecture and its parameters.
        dataset (DatasetConfig): Configuration for the dataset and its preprocessing.
        criterion (CriterionConfig): Configuration for the loss function.
        optimizer (OptimizerConfig | None): Configuration for the optimizer (optional, can be None in test-only mode).
        scheduler (SchedulerConfig | None): Configuration for the learning rate scheduler (optional, can be None in test-only mode).
        metric_tracker (MetricTrackerConfig): Configuration for tracking performance metrics.
        train (TrainConfig): Configuration for training settings.
        test (TestConfig): Configuration for testing settings.
        logger (LoggerConfig): Configuration for logging settings.

    Examples:
        >>> config = Config(
        ...     experiment=ExperimentConfig(
        ...         name="some_name",
        ...         paths={
        ...             "root_path": "some_experiment",
        ...             "model_path": "models/model.pth",
        ...             "best_model_path": "models/best_model.pth",
        ...             "optimizer_path": "optim/optimizer.pth",
        ...             "best_optimizer_path": "optim/best_optimizer.pth",
        ...             "train_metrics_path": "metrics/train_metrics.csv",
        ...             "val_metrics_path": "metrics/val_metrics.csv",
        ...             "test_metrics_path": "metrics/test_metrics.csv",
        ...             "graph_path": "progress.png",
        ...             "config_snapshot": "config_snapshot.yaml",
        ...             "logs_path": "logs.log"
        ...         }
        ...     ),
        ...     logger=LoggerConfig(level="DEBUG"),
        ...     train=TrainConfig(enabled=True, epochs=50),
        ...     test=TestConfig(enabled=True),
        ...     dataset=DatasetConfig(
        ...         obj="src.data.datasets.ImageClassificationDataset",
        ...         init_params={"path": "datasets/fruits_classification", "transforms": {
        ...             "obj": "torchvision.models.ResNet101_Weights.DEFAULT.transforms",
        ...             "init_params": {}
        ...         }},
        ...         post_params={"batch_size": 32}
        ...     ),
        ...     model=ModelConfig(
        ...         obj="torchvision.models.resnet101",
        ...         init_params={"weights": {
        ...             "obj": "torchvision.models.ResNet101_Weights.DEFAULT",
        ...             "init_params": None
        ...         }},
        ...         post_params={"out_features": 12, "freeze_pretrained_weights": True}
        ...     ),
        ...     criterion=CriterionConfig(obj=torch.nn.CrossEntropyLoss, init_params={}),
        ...     optimizer=OptimizerConfig(obj=torch.optim.Adam, init_params={}),
        ...     scheduler=SchedulerConfig(obj="torch.optim.lr_scheduler.ReduceLROnPlateau", init_params={}),
        ...     metric_tracker=MetricTrackerConfig(
        ...         obj="torchmetrics.Accuracy",
        ...         init_params={"task": "multiclass", "num_classes": 12}
        ...     )
        ... )
        >>> config  # doctest: +ELLIPSIS
        Config(experiment=ExperimentConfig(name='some_name', ...), ...



    Notes:
        - Fields marked as `| None` (e.g., `optimizer`, `scheduler`) are optional and can be omitted when only testing is required.
        - The `obj` and `init_params` pattern suggests compatibility with an object factory that resolves string paths
          into Python objects (e.g., `torch.nn.CrossEntropyLoss`).
        - Nested configurations (e.g., `model.init_params.weights`) allow for recursive object instantiation.
    """

    experiment: ExperimentConfig
    model: ModelConfig
    dataset: DatasetConfig
    criterion: CriterionConfig
    optimizer: OptimizerConfig | None = None  # if only test mode can be None
    scheduler: SchedulerConfig | None = None  # if only test mode can be None
    metric_tracker: MetricTrackerConfig
    train: TrainConfig
    test: TestConfig
    logger: LoggerConfig
