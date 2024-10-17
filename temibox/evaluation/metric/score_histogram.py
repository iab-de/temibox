import numpy as np
from typing import Any
from pandas import DataFrame

from .metric import Metric
from ...domain import UseCase, Document
from ...prediction import RawPrediction


class ScoreHistogram(Metric):
    r"""
    Calculates score histogram

    Compatible with binary classifiers
    """

    def __init__(self, bins: int = 10):
        self._bins = bins

    @property
    def name(self) -> str:
        return f"score-histogram ({self._bins} bins)"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:

        scores = [pi.score for p in raw_predictions for pi in p]

        min_score = min(scores)
        max_score = max(scores)
        bin_size = (max_score - min_score)/self._bins
        buckets = [[] for _ in range(self._bins)]
        thresholds = np.asarray([(min_score + bin_size*i) for i in range(1, self._bins+1)])
        for s in scores:
            buckets[np.argmax(s <= thresholds)].append(s)

        if return_dataframe:
            result = {f"({min_score+i*bin_size:.2f} - {min_score+(i+1)*bin_size:.2f}]": len(b)/len(scores) for i, b in enumerate(buckets)}
            df = DataFrame({"bin": result.keys(),
                            "relative_frequency": result.values()})
            return df

        hist = [f"({min_score+i*bin_size:.2f} - {min_score+(i+1)*bin_size:.2f}]: {len(b)/len(scores):.2%}" for i, b in enumerate(buckets)]
        return "\n".join(hist)