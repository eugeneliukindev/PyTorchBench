from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class ExperimentPaths(BaseModel):
    root_path: Path
    model_path: Path
    best_model_path: Path
    optimizer_path: Path
    best_optimizer_path: Path
    train_metrics_path: Path
    test_metrics_path: Path
    val_metrics_path: Path
    graph_path: Path
    config_snapshot: Path
    logs_path: Path

    @model_validator(mode="after")
    def prepend_base_path(self) -> "ExperimentPaths":
        for field in self.model_fields_set:
            (
                setattr(self, field, self.root_path / getattr(self, field))
                if field != "root_path"
                else None
            )
        return self


class DatasetPostParams(BaseModel):
    batch_size: Annotated[int, Field(ge=1)]


class ModelPostParams(BaseModel):
    out_features: Annotated[int, Field(ge=1)]
    freeze_pretrained_weights: bool
