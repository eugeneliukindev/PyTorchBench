from contextlib import nullcontext as does_not_raise
from typing import Annotated

import pytest
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Metric

from src.training import run_train
from tests.test_training.conftest import assert_engine_metrics


@pytest.mark.parametrize(
    "progress, epochs, starting_epoch, scheduler, expectation",
    [
        (True, 1, 1, "valid_scheduler", does_not_raise()),
        (False, 1, 1, "valid_scheduler", does_not_raise()),
        ("not_a_bool", 1, 1, "valid_scheduler", pytest.raises(TypeError, match="progress must be a boolean")),
        (True, 0, 1, "valid_scheduler", pytest.raises(ValueError, match="epochs must be a positive integer")),
        (True, -1, 1, "valid_scheduler", pytest.raises(ValueError, match="epochs must be a positive integer")),
        (
            True,
            "not_an_int",
            0,
            "valid_scheduler",
            pytest.raises(ValueError, match="epochs must be a positive integer"),
        ),
        (
            True,
            1,
            -1,
            "valid_scheduler",
            pytest.raises(ValueError, match="starting_epoch must be an integer and greater than or equal to 1"),
        ),
        (
            True,
            1,
            "not_an_int",
            "valid_scheduler",
            pytest.raises(ValueError, match="starting_epoch must be an integer and greater than or equal to 1"),
        ),
        (
            True,
            1,
            1,
            "not_a_scheduler",
            pytest.raises(TypeError, match="scheduler must be a LRScheduler, ReduceLROnPlateau or StepLR"),
        ),
    ],
)
def test_run_train(
    sample_model: Module,
    sample_criterion: Module,
    sample_optimizer: Optimizer,
    sample_scheduler: ReduceLROnPlateau,
    sample_metric: Metric,
    sample_dataloader: DataLoader,
    progress: bool,
    epochs: Annotated[int, "> 0"],
    starting_epoch: Annotated[int, ">= 0"],
    scheduler: str,
    expectation,
):
    actual_scheduler = sample_scheduler if scheduler == "valid_scheduler" else scheduler
    with expectation:
        train_gen = run_train(
            model=sample_model,
            starting_epoch=starting_epoch,
            epochs=epochs,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            accuracy_metrics=sample_metric,
            scheduler=actual_scheduler,
            progress=progress,
        )
        train_metrics, val_metrics = next(train_gen)
        assert_engine_metrics(train_metrics)
        assert_engine_metrics(val_metrics)
