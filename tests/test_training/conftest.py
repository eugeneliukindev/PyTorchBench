from unittest.mock import Mock

import pytest
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from src.training import run_test, EngineMetrics
from src.training.engine import run_engine_step


@pytest.fixture(scope="package")
def sample_model():
    model = Mock(spec=Module)
    model.train = Mock()
    model.eval = Mock()
    model.return_value = torch.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    return model


@pytest.fixture(scope="package")
def sample_criterion():
    criterion = Mock(spec=Module)
    criterion.return_value = torch.tensor(0.4, requires_grad=True)
    return criterion


@pytest.fixture(scope="package")
def sample_optimizer():
    optimizer = Mock(spec=Optimizer)
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    return optimizer


@pytest.fixture(scope="package")
def sample_scheduler():
    scheduler = Mock(spec=LRScheduler)
    scheduler.step = Mock()
    return scheduler


@pytest.fixture(scope="package")
def sample_metric():
    metric = Mock(spec=Metric)
    metric.update = Mock()
    metric.compute = Mock(return_value=torch.tensor(0.8))
    metric.reset = Mock()
    return metric


@pytest.fixture(scope="package")
def sample_dataloader():
    inputs = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    labels = torch.tensor([0, 1])
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture(scope="package")
def invalid_iterable():
    return [1, 2, 3]


@pytest.fixture(scope="package")
def tqdm_invalid_iterable():
    return tqdm([1, 2, 3])


@pytest.fixture(scope="package")
def invalid_model():
    return "not_a_module"  # Простая строка вместо объекта Module


@pytest.fixture(scope="package")
def invalid_criterion():
    return 42  # Число вместо объекта Module


@pytest.fixture(scope="package")
def invalid_metric():
    return ["list", "instead", "of", "metric"]  # Список вместо объекта Metric


def assert_engine_metrics(engine_metrics):
    assert isinstance(engine_metrics, EngineMetrics)
    assert engine_metrics.loss == pytest.approx(0.4)  # 0.4 * 2 samples / 2 (length of dataloader)
    assert engine_metrics.accuracy == pytest.approx(0.8)
    assert engine_metrics.time >= 0.0
