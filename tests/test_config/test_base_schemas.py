from typing import Any, ContextManager

import pytest
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from src.config.base_schemas import ObjectFactory


class TestObjectFactory:
    def test_create_obj_with_nested_mapping(self, config_dict):
        config = ObjectFactory(**config_dict["model"])
        assert config.obj == torchvision.models.resnet101
        assert config.init_params is not None
        assert config.init_params["weights"] == torchvision.models.ResNet101_Weights.DEFAULT
        assert isinstance(config.create(), torchvision.models.ResNet)

    def test_create_obj_with_nested_mapping_and_nested_sequence(self, config_dict):
        transform_config = ObjectFactory(**config_dict["dataset"]["init_params"]["transforms"])
        transform: v2.Compose = transform_config.create()
        assert isinstance(transform, v2.Compose)
        assert len(transform.transforms) == 5
        assert isinstance(transform.transforms[0], v2.RGB)
        assert isinstance(transform.transforms[1], v2.Resize)
        assert isinstance(transform.transforms[2], v2.ToImage)
        assert isinstance(transform.transforms[3], v2.ToDtype)
        assert isinstance(transform.transforms[4], nn.Sequential)
        sequential: nn.Sequential = transform.transforms[4]
        assert len(sequential) == 2
        assert isinstance(sequential[0], nn.Conv2d)
        assert isinstance(sequential[1], nn.ReLU)
        to_dtype_transform: v2.ToDtype = transform.transforms[3]
        assert to_dtype_transform.dtype == torch.float32
        assert to_dtype_transform.scale

    def test_create_obj_with_additional_params(self):
        config = ObjectFactory(obj="torch.nn.Conv2d", init_params={"in_channels": 3, "out_channels": 16})
        obj = config.create(kernel_size=3)
        assert isinstance(obj, nn.Conv2d)
        assert obj.kernel_size == (3, 3)

    @pytest.mark.parametrize(
        "obj, init_params, expectation",
        [
            (None, {}, pytest.raises(ValueError, match="init_params must be None when obj is not callable")),
            (42, {}, pytest.raises(ValueError, match="init_params must be None when obj is not callable")),
            (
                nn.CrossEntropyLoss,
                "not_a_dict",
                pytest.raises(ValueError, match="init_params must be a Mapping when obj is callable"),
            ),
        ],
        ids=[
            # success cases
            "none_obj__no_init_params",
            "int_obj__no_init_params",
            # exception cases
            "callable_obj__non_dict_init_params",
        ],
    )
    def test_obj_factory_exc(self, obj, init_params, expectation):
        with expectation:
            ObjectFactory(obj=obj, init_params=init_params)
