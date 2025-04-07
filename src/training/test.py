import logging
from typing import Any, TypeVar

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from tqdm import tqdm

from .engine import run_engine_step
from .metrics import EngineMetrics

log = logging.getLogger(__name__)

T = TypeVar("T")


@torch.no_grad()
def run_test(
    model: Module,
    dataloader: DataLoader[T],
    criterion: Module,
    accuracy_metrics: Metric,
    progress: bool = True,
) -> EngineMetrics:
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean")
    log.info("Starting testing")
    iterable: tqdm[DataLoader[T]] | DataLoader[T] = (
        tqdm(
            dataloader,
            total=len(dataloader),
            desc="Test",
            leave=True,
            colour="white",
        )
        if progress
        else dataloader
    )
    engine_metrics: EngineMetrics = run_engine_step(
        model=model,
        criterion=criterion,
        accuracy_metrics=accuracy_metrics,
        is_training=False,
        iterable=iterable,
    )
    log.info(
        "Test loss %.4f, Test accuracy %.4f, Test time %.4fs",
        engine_metrics.loss,
        engine_metrics.accuracy,
        engine_metrics.time,
    )
    return engine_metrics
