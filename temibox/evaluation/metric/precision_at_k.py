import numpy as np
from typing import Any
from pandas import DataFrame

from .metric import Metric
from ...domain import UseCase, Document
from ...prediction import RawPrediction


class PrecisionAtK(Metric):
    r"""
    Calculates precision-at-k, i.e. quotient of predicted relevant labels to number of predicted labels

    Compatible with multicategorical classifiers
    """
    def __init__(self, k: int = 10, min_score: float = 0.5):
        self._k = k
        self._min_score = min_score

    @property
    def name(self) -> str:
        return f"precision@k (k={self._k}, min-score={self._min_score:.2f})"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:

        if len(raw_predictions) != len(documents):
            raise Exception("Number of predictions does not match the number of documents")

        matches = [0.0] * len(documents)
        matches_per_len = {}
        for i, doc in enumerate(documents):
            tlabels = set(usecase.get_document_label_ids(doc))
            slabels = sorted(raw_predictions[i], key=lambda x: x.score, reverse=True)
            plabels = set([p.label_id for p in slabels[:self._k] if p.score >= self._min_score])

            matches[i] = len(plabels & tlabels)/max(1, min(self._k, len(plabels)))
            if len(tlabels) not in matches_per_len:
                matches_per_len[len(tlabels)] = []

            matches_per_len[len(tlabels)].append(matches[i])

        result = {"mean": np.mean(matches),
                  "mean_per_true_count": {k: np.mean(v) for k,v in matches_per_len.items()}}

        if return_dataframe:
            df = DataFrame({"label_count": result["mean_per_true_count"].keys(),
                            "precision_at_k": result["mean_per_true_count"].values(),
                            "overall": result["mean"]}) \
                    .sort_values("label_count") \
                    .reset_index(drop=True)

            return df

        return result