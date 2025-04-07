from contextlib import nullcontext as does_not_raise

import pytest

from src.training.engine import run_engine_step
from tests.test_training.conftest import assert_engine_metrics


class TestTrainingEngine:
    @pytest.mark.parametrize(
        "is_training, optimizer, iterable, model, criterion, accuracy_metrics, expectation",
        [
            (
                True,
                "sample_optimizer",
                "sample_dataloader",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                does_not_raise(),
            ),
            (
                False,
                "sample_optimizer",
                "sample_dataloader",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                does_not_raise(),
            ),
            (False, None, "sample_dataloader", "sample_model", "sample_criterion", "sample_metric", does_not_raise()),
            (
                True,
                None,
                "sample_dataloader",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                pytest.raises(ValueError, match="Optimizer is required for training"),
            ),
            (
                True,
                "sample_optimizer",
                "tqdm_invalid_iterable",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                pytest.raises(TypeError, match="tqdm.iterable is not a DataLoader"),
            ),
            (
                True,
                "sample_optimizer",
                "sample_dataloader",
                "invalid_model",
                "sample_criterion",
                "sample_metric",
                pytest.raises(TypeError, match="Expected Module"),
            ),
            (
                True,
                "sample_optimizer",
                "sample_dataloader",
                "sample_model",
                "invalid_criterion",
                "sample_metric",
                pytest.raises(TypeError, match="Expected Module"),
            ),
            (
                True,
                "sample_optimizer",
                "sample_dataloader",
                "sample_model",
                "sample_criterion",
                "invalid_metric",
                pytest.raises(TypeError, match="Expected Metric"),
            ),
            (
                True,
                "sample_optimizer",
                "invalid_iterable",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                pytest.raises(TypeError, match="Expected DataLoader or tqdm"),
            ),
            (
                "not_a_bool",
                "sample_optimizer",
                "sample_dataloader",
                "sample_model",
                "sample_criterion",
                "sample_metric",
                pytest.raises(TypeError, match="Expected bool"),
            ),
        ],
        ids=[
            "training_with_optimizer",
            "evaluation_with_optimizer",
            "evaluation_without_optimizer",
            "training_without_optimizer",
            "tqdm_invalid_iterable",
            "invalid_model",
            "invalid_criterion",
            "invalid_metric",
            "invalid_iterable",
            "invalid_is_training",
        ],
    )
    def test_run_engine_step(
        self,
        request,
        sample_model,
        sample_criterion,
        sample_optimizer,
        sample_metric,
        sample_dataloader,
        is_training,
        optimizer,
        iterable,
        model,
        criterion,
        accuracy_metrics,
        expectation,
    ):
        iterable_obj = request.getfixturevalue(iterable) if isinstance(iterable, str) else iterable
        model_obj = request.getfixturevalue(model) if isinstance(model, str) else model
        criterion_obj = request.getfixturevalue(criterion) if isinstance(criterion, str) else criterion
        accuracy_metrics_obj = (
            request.getfixturevalue(accuracy_metrics) if isinstance(accuracy_metrics, str) else accuracy_metrics
        )
        optimizer_obj = request.getfixturevalue(optimizer) if isinstance(optimizer, str) else optimizer

        with expectation:
            engine_metrics = run_engine_step(
                model=model_obj,
                criterion=criterion_obj,
                accuracy_metrics=accuracy_metrics_obj,
                iterable=iterable_obj,
                is_training=is_training,
                optimizer=optimizer_obj,
            )
            assert_engine_metrics(engine_metrics)
