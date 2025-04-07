from contextlib import nullcontext as does_not_raise

import pytest
import torch.nn as nn
from src.models import prepare_model
from tests.conftest import resnet18_model


@pytest.mark.parametrize(
    "out_features, freeze_pretrained_weights, expectation",
    [
        (10, True, does_not_raise()),
        (999, False, does_not_raise()),
        (-1, False, pytest.raises(ValueError, match="out_features must be greater than 0")),
    ],
    ids=[
        # success cases
        "normal_freeze",
        "large_no_freeze",
        # exception cases
        "negative_out_features",
    ],
)
def test_prepare_model(resnet18_model, out_features: int, freeze_pretrained_weights: bool, expectation):
    with expectation:
        model = prepare_model(
            resnet18_model, out_features=out_features, freeze_pretrained_weights=freeze_pretrained_weights
        )
        assert model.fc.out_features == out_features
        assert str(model.fc) == str(nn.Linear(in_features=model.fc.in_features, out_features=out_features))
