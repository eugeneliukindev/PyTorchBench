import logging
from typing import Generator, Annotated

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm
from typing_extensions import TypeVar

from .engine import run_engine_step
from .metrics import TrainMetrics, ValidationMetrics

log = logging.getLogger(__name__)

T = TypeVar("T")

type SchedulerType = LRScheduler | ReduceLROnPlateau | StepLR


def _validate_train(
    epochs: Annotated[int, "> 0"],
    starting_epoch: Annotated[int, ">= 0"],
    scheduler: SchedulerType,
    progress: bool | None = True,
) -> None:
    if not isinstance(epochs, int) or not epochs > 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(starting_epoch, int) or not starting_epoch >= 1:
        raise ValueError("starting_epoch must be an integer and greater than or equal to 1")
    if not isinstance(scheduler, (LRScheduler, ReduceLROnPlateau, StepLR)):
        raise TypeError("scheduler must be a LRScheduler, ReduceLROnPlateau or StepLR")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean")


def run_train(
    model: Module,
    epochs: Annotated[int, "> 0"],
    train_dataloader: DataLoader[T],
    val_dataloader: DataLoader[T],
    optimizer: Optimizer,
    criterion: Module,
    accuracy_metrics: Metric,
    scheduler: SchedulerType,
    starting_epoch: Annotated[int, ">= 1"] = 1,
    progress: bool = True,
) -> Generator[tuple[TrainMetrics, ValidationMetrics], None, None]:
    _validate_train(
        epochs=epochs,
        starting_epoch=starting_epoch,
        scheduler=scheduler,
        progress=progress,
    )
    log.info("Starting training")
    for epoch in range(starting_epoch, epochs + 1):
        train_iterable: tqdm[DataLoader[T]] | DataLoader[T] = (
            tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Epoch {epoch}/{epochs} Train",
                leave=True,
                colour="white",
            )
            if progress
            else train_dataloader
        )
        train_engine_metrics = run_engine_step(
            model=model,
            criterion=criterion,
            accuracy_metrics=accuracy_metrics,
            is_training=True,
            optimizer=optimizer,
            iterable=train_iterable,
        )
        train_metrics = TrainMetrics(epoch=epoch, **train_engine_metrics.model_dump())

        val_iterable: tqdm[DataLoader[T]] | DataLoader[T] = (
            tqdm(
                val_dataloader,
                total=len(val_dataloader),
                desc=f"Epoch {epoch + 1}/{epochs} Validate",
                leave=True,
                colour="white",
            )
            if progress
            else val_dataloader
        )
        val_engine_metrics = run_engine_step(
            model=model,
            criterion=criterion,
            accuracy_metrics=accuracy_metrics,
            is_training=False,
            iterable=val_iterable,
        )
        val_metrics = ValidationMetrics(epoch=epoch, **val_engine_metrics.model_dump())

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics.loss)
        elif isinstance(scheduler, StepLR):
            scheduler.step(epoch)
        else:
            scheduler.step()

        log.info(
            "Epoch %d completed: "
            "Train Loss: %.4f, "
            "Train Acc: %.4f, "
            "Val Loss: %.4f, "
            "Val Acc: %.4f, "
            "Train time: %.2fs, "
            "Val time: %.2fs",
            epoch,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
            train_metrics.time,
            val_metrics.time,
        )

        yield train_metrics, val_metrics
