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
    def validate_and_prepend_paths(self) -> "ExperimentPaths":
        if self.root_path.suffix:
            raise ValueError("Root path cannot have a suffix.")

        suffix_rules = {
            ".pth": (self.model_path, self.best_model_path, self.optimizer_path, self.best_optimizer_path),
            ".csv": (self.train_metrics_path, self.val_metrics_path, self.test_metrics_path),
            (".png", ".jpg", ".jpeg"): (self.graph_path,),
            ".yaml": (self.config_snapshot,),
            ".log": (self.logs_path,),
        }

        invalid_paths = []
        for expected_suffixes, paths in suffix_rules.items():
            expected_set = expected_suffixes if isinstance(expected_suffixes, tuple) else (expected_suffixes,)
            for path in paths:
                if path.suffix not in expected_set:
                    invalid_paths.append(f"{path}, expected: {expected_set}")

        if invalid_paths:
            raise ValueError(f"Invalid path suffixes: {'; '.join(invalid_paths)}")

        for field in self.model_fields_set:
            if field != "root_path":
                setattr(self, field, self.root_path / getattr(self, field))

        return self


class DatasetPostParams(BaseModel):
    batch_size: Annotated[int, Field(ge=1)]


class ModelPostParams(BaseModel):
    out_features: Annotated[int, Field(ge=1)]
    freeze_pretrained_weights: bool
