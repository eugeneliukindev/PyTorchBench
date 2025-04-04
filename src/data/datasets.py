import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import TypeVar, Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

type TensorDataset = Dataset[tuple[Tensor, ...]]
type TensorDataloader = DataLoader[TensorDataset]

type ClassLabel = int
type ClassLabelTensor = torch.LongTensor
type ImageFilePath = str
type ImageClassificationOutput = tuple[Tensor, Tensor]

log = logging.getLogger(__name__)

_T_co = TypeVar("_T_co", covariant=True)


class AbstractDataset(Dataset[_T_co], ABC):
    def __init__(
        self,
        path: str | PathLike[str],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if not isinstance(path, (str, PathLike)):
            raise TypeError(f"Expected str or PathLike, got {type(path).__name__}")
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device).__name__}")
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"path {self.path} does not exist.")
        self.device = device
        self._data: list[Any] = []
        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> _T_co:
        pass

    def __len__(self) -> int:
        return len(self._data)


class ImageClassificationDataset(AbstractDataset[ImageClassificationOutput]):
    def __init__(
        self,
        path: str | PathLike[str] | Path,
        device: torch.device,
        transforms: Module,
    ) -> None:
        super().__init__(path=path, device=device)
        if not isinstance(transforms, Module):
            raise TypeError(f"Expected Module, got {type(transforms).__name__}")
        self.transforms = transforms

    def _load_data(self) -> None:
        dir_content_paths = sorted(self.path.iterdir())
        class_dirs_list = [d for d in dir_content_paths if d.is_dir()]
        self.class_dirs: dict[int, str] = {
            i: d.name for i, d in enumerate(class_dirs_list)
        }

        self._data: list[tuple[int, str]] = []
        for idx, class_name in self.class_dirs.items():
            class_path = self.path / class_name
            for image_path in class_path.iterdir():
                if image_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self._data.append((idx, str(image_path)))

    def __getitem__(self, index: int) -> ImageClassificationOutput:
        class_index, image_path = self._data[index]
        image: Image.Image = Image.open(image_path).convert("RGB")
        tensor_image: Tensor = self.transforms(image)
        if not isinstance(tensor_image, torch.Tensor):
            raise TypeError(
                f"Transforms must return torch.Tensor, got {type(tensor_image).__name__}"
            )
        tensor_image = tensor_image.to(self.device)
        tensor_class: Tensor = torch.tensor(class_index, dtype=torch.long).to(
            self.device
        )

        return tensor_image, tensor_class
