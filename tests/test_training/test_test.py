import pytest

from src.training import run_test, EngineMetrics
from contextlib import nullcontext as does_not_raise

from tests.test_training.conftest import assert_engine_metrics


@pytest.mark.parametrize(
    "progress, expectation",
    [
        (True, does_not_raise()),
        (False, does_not_raise()),
        (None, pytest.raises(TypeError, match="progress must be a boolean")),
    ],
)
def test_run_test(sample_model, sample_criterion, sample_metric, sample_dataloader, progress, expectation):
    with expectation:
        engine_metrics = run_test(
            model=sample_model,
            criterion=sample_criterion,
            accuracy_metrics=sample_metric,
            dataloader=sample_dataloader,
            progress=progress,
        )
        assert_engine_metrics(engine_metrics)
