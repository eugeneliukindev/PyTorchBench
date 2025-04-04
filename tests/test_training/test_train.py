import pytest
from src.training import run_train, EngineMetrics
from contextlib import nullcontext as does_not_raise
from tests.test_training.conftest import assert_engine_metrics
from typing import Annotated
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, StepLR
from torchmetrics import Metric
from src.data import TensorDataloader


@pytest.mark.parametrize(
    "progress, epochs, start_epoch, scheduler, expectation",
    [
        (True, 1, 0, "valid_scheduler", does_not_raise()),
        (False, 1, 0, "valid_scheduler", does_not_raise()),
        ("not_a_bool", 1, 0, "valid_scheduler", pytest.raises(TypeError, match="progress must be a boolean")),
        (True, 0, 0, "valid_scheduler", pytest.raises(ValueError, match="epochs must be a positive integer")),
        (True, -1, 0, "valid_scheduler", pytest.raises(ValueError, match="epochs must be a positive integer")),
        (
            True,
            "not_an_int",
            0,
            "valid_scheduler",
            pytest.raises(ValueError, match="epochs must be a positive integer"),
        ),
        (True, 1, -1, "valid_scheduler", pytest.raises(ValueError, match="start_epoch must be a positive integer")),
        (
            True,
            1,
            "not_an_int",
            "valid_scheduler",
            pytest.raises(ValueError, match="start_epoch must be a positive integer"),
        ),
        (
            True,
            1,
            0,
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
    sample_dataloader: TensorDataloader,
    progress: bool,
    epochs: Annotated[int, "> 0"],
    start_epoch: Annotated[int, ">= 0"],
    scheduler: str,
    expectation,
):
    actual_scheduler = sample_scheduler if scheduler == "valid_scheduler" else scheduler
    with expectation:
        train_gen = run_train(
            model=sample_model,
            start_epoch=start_epoch,
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
