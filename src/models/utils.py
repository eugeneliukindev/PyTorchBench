from typing import Annotated, TypeVar

from torch import nn as nn
from torchvision.models import ResNet, VGG

from src.utils.general import log

T = TypeVar("T", bound=nn.Module)


def _validate_pretrained_model(
    model: nn.Module,
    out_features: Annotated[int, "> 0"],
    freeze_pretrained_weights: bool,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model).__name__}")
    if not out_features > 0:
        raise ValueError("out_features must be greater than 0")
    if not isinstance(freeze_pretrained_weights, bool):
        raise TypeError(f"Expected bool, got {type(freeze_pretrained_weights).__name__}")


def prepare_model(
    model: T,
    out_features: Annotated[int, "> 0"],
    freeze_pretrained_weights: bool = True,
) -> T:
    _validate_pretrained_model(
        model=model,
        out_features=out_features,
        freeze_pretrained_weights=freeze_pretrained_weights,
    )
    model_type = type(model).__name__
    log.debug(
        "Preparing model: type=%s, out_features=%d, freeze_pretrained_weights=%s",
        model_type,
        out_features,
        freeze_pretrained_weights,
    )

    if freeze_pretrained_weights:
        model.requires_grad_(False)
        log.debug("Loaded pretrained weights and froze all layers for %s", model_type)

    if isinstance(model, ResNet):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=out_features).requires_grad_(True)
        log.debug("Replaced ResNet fc layer: in_features=%d, out_features=%d", in_features, out_features)
    elif isinstance(model, VGG):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=in_features, out_features=out_features).requires_grad_(True)
        log.debug("Replaced VGG classifier[-1] layer: in_features=%d, out_features=%d", in_features, out_features)
    else:
        log.warning("Model type %s not explicitly handled", model_type)

    log.info("Model %s prepared with %d output features", model_type, out_features)
    return model
