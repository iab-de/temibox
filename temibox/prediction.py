from dataclasses import dataclass, asdict

from typing import Dict, Any
from pandas import DataFrame

from .domain import Label

@dataclass
class RawPrediction:
    r"""
    Representation of a raw prediction.

    Contains the label string, label id and prediction score
    """
    label: Label
    label_id: int
    score: float

@dataclass
class Prediction:
    r"""
    Representation of a generic container for prediction results (payload)

    This data type is (recommended) by the Predictable trait
    """
    usecase_name: str
    model: str
    timestamp: int
    payload_raw: list[RawPrediction]
    payload: DataFrame | Any

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Converts a dataclass to a dictionary (wrapper for dataclasses.asdict)

        :return: dict
        """
        return asdict(self)