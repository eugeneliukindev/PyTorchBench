from pydantic import BaseModel


class EngineMetrics(BaseModel):
    loss: float
    accuracy: float
    time: float


class EpochMetrics(EngineMetrics):
    epoch: int


class TrainMetrics(EpochMetrics):
    pass


class ValidationMetrics(EpochMetrics):
    pass


class TestMetrics(EpochMetrics):
    pass
