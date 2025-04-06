import argparse
import logging
from typing import Any, cast, TypeVar

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics import Metric

from src.config import setup_experiment, OptimizerConfig, SchedulerConfig
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

_T = TypeVar("_T")

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_split_dataloaders(
    dataset: Dataset[_T], batch_size: int
) -> tuple[DataLoader[_T], DataLoader[_T], DataLoader[_T]]:
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    config = setup_experiment(config_path)
    dataset: Dataset[Any] = config.dataset.create(expected_obj=Dataset, device=DEVICE)

    model: Module = prepare_model(
        model=config.model.create(expected_obj=Module),
        out_features=config.model.post_params.out_features,
        freeze_pretrained_weights=config.model.post_params.freeze_pretrained_weights,
    ).to(DEVICE)
    criterion: Module = config.criterion.create(expected_obj=Module)
    metric_tracker: Metric = config.metric_tracker.create(expected_obj=Metric).to(DEVICE)  # type: ignore[type-abstract]
    train_dataloader, val_dataloader, sample_dataloader = _get_split_dataloaders(
        dataset=dataset, batch_size=config.dataset.post_params.batch_size
    )

    if config.train.enabled:
        optimizer: Optimizer = cast(OptimizerConfig, config.optimizer).create(
            expected_obj=Optimizer, params=model.parameters()
        )
        scheduler: LRScheduler = cast(SchedulerConfig, config.scheduler).create(
            expected_obj=LRScheduler, optimizer=optimizer
        )

        if all(
            i.exists()
            for i in (
                config.experiment.paths.model_path,
                config.experiment.paths.best_model_path,
                config.experiment.paths.optimizer_path,
                config.experiment.paths.train_metrics_path,
                config.experiment.paths.val_metrics_path,
            )
        ):
            model.load_state_dict(torch.load(config.experiment.paths.model_path))
            optimizer.load_state_dict(torch.load(config.experiment.paths.optimizer_path))
            train_metrics_history = [
                TrainMetrics(**data) for data in load_from_csv(config.experiment.paths.train_metrics_path)
            ]
            val_metrics_history = [
                ValidationMetrics(**data) for data in load_from_csv(config.experiment.paths.val_metrics_path)
            ]
            best_validation_loss = min(val_metrics_history, key=lambda m: m.loss).loss
            starting_epoch = max(train_metrics_history, key=lambda m: m.epoch).epoch
        else:
            train_metrics_history = []
            val_metrics_history = []
            best_validation_loss = float("inf")
            starting_epoch = 0

        for train_epoch_metrics, val_epoch_metrics in run_train(
            model=model,
            start_epoch=starting_epoch,
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

            if val_epoch_metrics.loss < best_validation_loss:
                best_validation_loss = val_epoch_metrics.loss
                torch.save(model.state_dict(), config.experiment.paths.best_model_path)
                torch.save(optimizer.state_dict(), config.experiment.paths.best_optimizer_path)
            torch.save(model.state_dict(), config.experiment.paths.model_path)
            torch.save(optimizer.state_dict(), config.experiment.paths.optimizer_path)
            save_to_csv(
                data=train_epoch_metrics.model_dump(),
                path=config.experiment.paths.train_metrics_path,
            )
            save_to_csv(
                data=val_epoch_metrics.model_dump(),
                path=config.experiment.paths.val_metrics_path,
            )
            save_graph(
                train_metrics_history=train_metrics_history,
                val_metrics_history=val_metrics_history,
                path=config.experiment.paths.graph_path,
            )

    if config.test.enabled:
        if not config.experiment.paths.best_model_path.exists():
            raise ValueError("Best model weights not found; training is required first")
        model.load_state_dict(torch.load(config.experiment.paths.best_model_path))
        val_metrics_history = [
            ValidationMetrics(**data) for data in load_from_csv(config.experiment.paths.val_metrics_path)
        ]
        best_epoch = min(val_metrics_history, key=lambda m: m.loss).epoch
        log.info("Testing on epoch %d", best_epoch)
        test_run_metrics: EngineMetrics = run_test(
            model=model,
            dataloader=sample_dataloader,
            criterion=criterion,
            accuracy_metrics=metric_tracker,
        )
        test_results: TestMetrics = TestMetrics(epoch=best_epoch, **test_run_metrics.model_dump())
        save_to_csv(
            data=test_results.model_dump(),
            path=config.experiment.paths.test_metrics_path,
        )


if __name__ == "__main__":
    main()
