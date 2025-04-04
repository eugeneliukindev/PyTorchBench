from copy import deepcopy
from pathlib import Path

import pytest
import torchvision

import src
from src.config.experiment import (
    _check_critical_component_changes,
    CRITICAL_COMPONENTS,
    CriticalComponent,
    setup_experiment,
)
from tests.conftest import NO_RESULT
from contextlib import nullcontext as does_not_raise


@pytest.mark.parametrize(
    "changes, critical_components, expectation",
    [
        ({}, CRITICAL_COMPONENTS["train"], does_not_raise()),
        ({}, CRITICAL_COMPONENTS["test"], does_not_raise()),
        # Optimizer changes not critical for test
        (
            {"optimizer": {"obj": "torch.optim.SGD"}},
            CRITICAL_COMPONENTS["test"],
            does_not_raise(),
        ),
        # Scheduler changes not critical for test
        (
            {"scheduler": {"obj": "torch.optim.lr_scheduler.ReduceLROnPlateau"}},
            CRITICAL_COMPONENTS["test"],
            does_not_raise(),
        ),
        # Model changes
        (
            {"model": {"obj": "torchvision.models.resnet50"}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        (
            {"model": {"init_params": {"weights": None}}},
            CRITICAL_COMPONENTS["test"],
            pytest.raises(ValueError),
        ),
        # Dataset changes
        (
            {"dataset": {"obj": "data.datasets.CustomDataset"}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        (
            {"dataset": {"init_params": {"path": "new_data/"}}},
            CRITICAL_COMPONENTS["test"],
            pytest.raises(ValueError),
        ),
        # Criterion changes
        (
            {"criterion": {"obj": "torch.nn.BCEWithLogitsLoss"}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        (
            {"criterion": {"init_params": {"reduction": "sum"}}},
            CRITICAL_COMPONENTS["test"],
            pytest.raises(ValueError),
        ),
        (
            {"optimizer": {"obj": "torch.optim.SGD"}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        # Optimizer changes
        (
            {"optimizer": {"init_params": {"lr": 0.01}}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        (
            {"scheduler": {"obj": "torch.optim.lr_scheduler.ReduceLROnPlateau"}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        # Scheduler changes
        (
            {"scheduler": {"init_params": {"step_size": 5}}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        # Nested changes
        (
            {"model": {"init_params": {"weights": {"obj": "new_weights"}}}},
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
        (
            {"dataset": {"init_params": {"transforms": {"obj": "new_transform"}}}},
            CRITICAL_COMPONENTS["test"],
            pytest.raises(ValueError),
        ),
        # Multiple changes
        (
            {
                "model": {"obj": "torchvision.models.resnet50"},
                "dataset": {"init_params": {"path": "new_path"}},
            },
            CRITICAL_COMPONENTS["train"],
            pytest.raises(ValueError),
        ),
    ],
    ids=[
        # success cases
        "no_changes_train",
        "no_changes_test",
        "optimizer_change_test",
        "scheduler_change_test",
        # exception cases
        "model_obj_change_train",
        "model_params_change_test",
        "dataset_obj_change_train",
        "dataset_params_change_test",
        "criterion_obj_change_train",
        "criterion_params_change_test",
        "optimizer_obj_change_train",
        "optimizer_params_change_train",
        "scheduler_obj_change_train",
        "scheduler_params_change_train",
        "nested_model_change_train",
        "nested_dataset_change_test",
        "multiple_changes_train",
    ],
)
def test_check_critical_component_changes(
    config_dict, changes, critical_components, expectation
):
    current_config = config_dict.copy()
    previous_config = config_dict.copy()

    previous_config.update(changes)

    with expectation:
        _check_critical_component_changes(
            current_config=current_config,
            previous_config=previous_config,
            critical_components=critical_components,
        )


def test_setup_experiment(tmp_path, config_yaml):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    config = setup_experiment(config_path)

    assert config.model.obj == torchvision.models.resnet101
    assert (
        config.model.init_params["weights"]
        == torchvision.models.ResNet101_Weights.DEFAULT
    )
    assert config.dataset.obj == src.data.datasets.ImageClassificationDataset
    assert config.train.enabled is True
