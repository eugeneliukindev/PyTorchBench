from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.additional_schemas import (
    ExperimentPaths,
    DatasetPostParams,
    ModelPostParams,
)


class TestExperimentPaths:

    def test_creates_paths_with_valid_data(self, config_dict):
        paths_dict = config_dict["experiment"]["paths"]
        paths = ExperimentPaths(**paths_dict)
        root_path = Path(paths_dict["root_path"])
        for field_name in paths.model_fields_set:
            value = getattr(paths, field_name)
            assert isinstance(value, Path), f"Field '{field_name}' is not a Path, got {type(value)}"

            if field_name == "root_path":
                assert value == root_path, f"root_path mismatch: {value} != {root_path}"
            else:
                expected_value = root_path / paths_dict[field_name]
                assert value == expected_value, f"{field_name} mismatch: {value} != {expected_value}"


class TestDatasetPostParams:
    @pytest.mark.parametrize(
        "batch_size, expectation",
        [
            (1, does_not_raise()),
            (32, does_not_raise()),
            (0, pytest.raises(ValidationError, match="greater than or equal to 1")),
            (-1, pytest.raises(ValidationError, match="greater than or equal to 1")),
        ],
        ids=[
            "batch_size_1",
            "batch_size_32",
            "batch_size_zero",
            "batch_size_negative",
        ],
    )
    def test_validates_batch_size(self, batch_size, expectation):
        with expectation:
            DatasetPostParams(batch_size=batch_size)


class TestModelPostParams:
    @pytest.mark.parametrize(
        "out_features, freeze_pretrained_weights, expectation",
        [
            (10, True, does_not_raise()),
            (999, False, does_not_raise()),
            (0, False, pytest.raises(ValidationError, match="greater than or equal to 1")),
            (-1, False, pytest.raises(ValidationError, match="greater than or equal to 1")),
        ],
        ids=[
            # success cases
            "out_features_10__freeze_true",
            "out_features_999__freeze_false",
            # exception cases
            "out_features_zero__freeze_false",
            "out_features_negative__freeze_false",
        ],
    )
    def test_creates_params_with_valid_data(self, out_features, freeze_pretrained_weights, expectation) -> None:
        with expectation:
            model_post_params = ModelPostParams(
                out_features=out_features, freeze_pretrained_weights=freeze_pretrained_weights
            )
            assert model_post_params.out_features == out_features
            assert model_post_params.freeze_pretrained_weights == freeze_pretrained_weights
