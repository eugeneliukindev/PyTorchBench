import logging
from typing import cast, TypeVar

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics import Metric

from src.config import setup_experiment, OptimizerConfig, SchedulerConfig, Config
from src.models import prepare_model
from src.training import (
    TrainMetrics,
    ValidationMetrics,
    run_train,
    run_test,
    EngineMetrics,
    TestMetrics,
)
from src.utils import load_from_csv, save_to_csv, save_graph

log = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_split_dataloaders(
    dataset: Dataset[T_co], batch_size: int
) -> tuple[DataLoader[T_co], DataLoader[T_co], DataLoader[T_co]]:
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def setup_model(config: Config) -> Module:
    model = config.model.create(expected_obj=Module)
    return prepare_model(
        model=model,
        out_features=config.model.post_params.out_features,
        freeze_pretrained_weights=config.model.post_params.freeze_pretrained_weights,
    )


def load_training_state(
    config: Config, model: Module, optimizer: Optimizer
) -> tuple[list[TrainMetrics], list[ValidationMetrics], float, int]:
    paths = (
        config.experiment.paths.model_path,
        config.experiment.paths.optimizer_path,
        config.experiment.paths.train_metrics_path,
        config.experiment.paths.val_metrics_path,
    )
    if all(path.exists() for path in paths):
        model.load_state_dict(torch.load(config.experiment.paths.model_path))
        optimizer.load_state_dict(torch.load(config.experiment.paths.optimizer_path))
        train_metrics_history = [
            TrainMetrics(**data) for data in load_from_csv(config.experiment.paths.train_metrics_path)
        ]
        val_metrics_history = [
            ValidationMetrics(**data) for data in load_from_csv(config.experiment.paths.val_metrics_path)
        ]
        best_val_loss = min(val_metrics_history, key=lambda m: m.loss).loss
        starting_epoch = max(train_metrics_history, key=lambda m: m.epoch).epoch + 1
    else:
        train_metrics_history = []
        val_metrics_history = []
        best_val_loss = float("inf")
        starting_epoch = 1
    return train_metrics_history, val_metrics_history, best_val_loss, starting_epoch


def save_training_progress(
    config: Config,
    model: Module,
    optimizer: Optimizer,
    train_epoch_metrics: TrainMetrics,
    val_epoch_metrics: ValidationMetrics,
    train_metrics_history: list[TrainMetrics],
    val_metrics_history: list[ValidationMetrics],
    best_val_loss: float,
) -> float:
    if val_epoch_metrics.loss < best_val_loss:
        best_val_loss = val_epoch_metrics.loss
        torch.save(model.state_dict(), config.experiment.paths.best_model_path)
        torch.save(optimizer.state_dict(), config.experiment.paths.best_optimizer_path)
    torch.save(model.state_dict(), config.experiment.paths.model_path)
    torch.save(optimizer.state_dict(), config.experiment.paths.optimizer_path)
    save_to_csv(data=train_epoch_metrics.model_dump(), path=config.experiment.paths.train_metrics_path)
    save_to_csv(data=val_epoch_metrics.model_dump(), path=config.experiment.paths.val_metrics_path)
    save_graph(
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
        path=config.experiment.paths.graph_path,
    )
    return best_val_loss


def load_testing_state(config: Config, model: Module) -> int:
    model.load_state_dict(torch.load(config.experiment.paths.best_model_path))
    val_metrics_history = [
        ValidationMetrics(**data) for data in load_from_csv(config.experiment.paths.val_metrics_path)
    ]
    best_epoch = min(val_metrics_history, key=lambda m: m.loss).epoch
    return best_epoch


def save_testing_progress(config: Config, test_metrics: TestMetrics) -> None:
    save_to_csv(
        data=test_metrics.model_dump(),
        path=config.experiment.paths.test_metrics_path,
    )


def start_training(
    config: Config,
    model: Module,
    criterion: Module,
    metric_tracker: Metric,
    train_dataloader: DataLoader[T_co],
    val_dataloader: DataLoader[T_co],
) -> None:
    optimizer: Optimizer = cast(OptimizerConfig, config.optimizer).create(
        expected_obj=Optimizer,
        params=model.parameters(),
    )
    scheduler: LRScheduler = cast(SchedulerConfig, config.scheduler).create(
        expected_obj=LRScheduler,
        optimizer=optimizer,
    )

    train_metrics_history, val_metrics_history, best_val_loss, starting_epoch = load_training_state(
        config=config,
        model=model,
        optimizer=optimizer,
    )

    for train_epoch_metrics, val_epoch_metrics in run_train(
        model=model,
        starting_epoch=starting_epoch,
        epochs=config.train.epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        accuracy_metrics=metric_tracker,
        scheduler=scheduler,
    ):
        train_metrics_history.append(train_epoch_metrics)
        val_metrics_history.append(val_epoch_metrics)

        best_val_loss = save_training_progress(
            config=config,
            model=model,
            optimizer=optimizer,
            train_epoch_metrics=train_epoch_metrics,
            val_epoch_metrics=val_epoch_metrics,
            best_val_loss=best_val_loss,
            train_metrics_history=train_metrics_history,
            val_metrics_history=val_metrics_history,
        )


def start_testing(
    config: Config,
    model: Module,
    criterion: Module,
    metric_tracker: Metric,
    test_dataloader: DataLoader[T_co],
) -> None:
    if not config.experiment.paths.best_model_path.exists():
        raise ValueError("Best model weights not found; training is required first")
    best_epoch = load_testing_state(config=config, model=model)
    log.info("Testing on epoch %d", best_epoch)
    engine_metrics: EngineMetrics = run_test(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        accuracy_metrics=metric_tracker,
    )
    test_metrics: TestMetrics = TestMetrics(epoch=best_epoch, **engine_metrics.model_dump())
    save_testing_progress(config=config, test_metrics=test_metrics)


def main() -> None:
    config = setup_experiment()

    dataset = config.dataset.create(expected_obj=Dataset, device=DEVICE)
    train_dataloader, val_dataloader, test_dataloader = _get_split_dataloaders(
        dataset=dataset,
        batch_size=config.dataset.post_params.batch_size,
    )

    model = setup_model(config).to(DEVICE)
    criterion: Module = config.criterion.create(expected_obj=Module)
    metric_tracker: Metric = config.metric_tracker.create(expected_obj=Metric).to(DEVICE)  # type: ignore[type-abstract]

    if config.train.enabled:
        start_training(
            config=config,
            model=model,
            criterion=criterion,
            metric_tracker=metric_tracker,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

    if config.test.enabled:
        start_testing(
            config=config,
            model=model,
            criterion=criterion,
            metric_tracker=metric_tracker,
            test_dataloader=test_dataloader,
        )


if __name__ == "__main__":
    main()
