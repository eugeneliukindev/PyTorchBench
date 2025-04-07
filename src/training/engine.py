import logging
import time
from typing import cast, TypeVar

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from .metrics import EngineMetrics

log = logging.getLogger(__name__)

T = TypeVar("T")


def _validate_engine(
    model: Module,
    criterion: Module,
    accuracy_metrics: Metric,
    iterable: tqdm | DataLoader[T],
    is_training: bool,
    optimizer: Optimizer | None = None,
) -> bool:
    if not isinstance(model, Module):
        raise TypeError(f"Expected Module, got {type(model).__name__}")
    if not isinstance(criterion, Module):
        raise TypeError(f"Expected Module, got {type(criterion).__name__}")
    if not isinstance(accuracy_metrics, Metric):
        raise TypeError(f"Expected Metric, got {type(accuracy_metrics).__name__}")
    if not isinstance(iterable, DataLoader) and not isinstance(iterable, tqdm):
        raise TypeError(f"Expected DataLoader or tqdm, got {type(iterable).__name__}")
    if not isinstance(optimizer, Optimizer) and optimizer is not None:
        raise TypeError(f"Expected Optimizer or None, got {type(optimizer).__name__}")
    if not isinstance(is_training, bool):
        raise TypeError(f"Expected bool, got {type(is_training).__name__}")

    tqdm_flag = False
    if isinstance(iterable, tqdm):
        if not isinstance(iterable.iterable, DataLoader):
            raise TypeError("tqdm.iterable is not a DataLoader")
        tqdm_flag = True

    if is_training and optimizer is None:
        raise ValueError("Optimizer is required for training")

    return tqdm_flag


def run_engine_step(
    model: Module,
    criterion: Module,
    accuracy_metrics: Metric,
    iterable: tqdm | DataLoader[T],
    is_training: bool,
    optimizer: Optimizer | None = None,
) -> EngineMetrics:
    tqdm_flag = _validate_engine(
        model=model,
        criterion=criterion,
        accuracy_metrics=accuracy_metrics,
        iterable=iterable,
        is_training=is_training,
        optimizer=optimizer,
    )
    log.info("Running engine step")
    start_time = time.time()
    total_loss = 0
    model.train() if is_training else model.eval()
    with torch.set_grad_enabled(is_training):
        for data, targets in iterable:
            if is_training:
                cast(Optimizer, optimizer).zero_grad()

            predictions = model(data)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                cast(Optimizer, optimizer).step()

            total_loss += loss.item()
            accuracy_metrics.update(predictions, targets)

            if tqdm_flag:
                cast(tqdm, iterable).set_postfix(loss=loss.item(), accuracy=accuracy_metrics.compute().item())

    total_time = time.time() - start_time
    avg_loss = total_loss / len(iterable)
    accuracy = accuracy_metrics.compute().item()
    accuracy_metrics.reset()

    return EngineMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        time=total_time,
    )
