from sklearn.metrics import f1_score, precision_score, recall_score
from pandas import DataFrame
from typing import Any

from ...domain import UseCase, Document
from ...prediction import RawPrediction
from .metric import Metric


class F1(Metric):
    r"""
    Calculates precision, recall and F1 scores

    Compatible with binary classifiers
    """

    def __init__(self, binary: bool = True):

        self._average = "binary" if binary else "macro"

    @property
    def name(self) -> str:
        return "f1"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase, documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:

        y    = [label for d in documents for label in usecase.get_document_label_ids(d)]
        yhat = [entry.label_id for rp in raw_predictions for entry in rp]

        assert len(y) == len(yhat), "True label count and prediction count don't match"

        score_f1        = f1_score(y_true = y, y_pred = yhat, average=self._average, zero_division=0.0)
        score_precision = precision_score(y_true = y, y_pred = yhat, average=self._average, zero_division=0.0)
        score_recall    = recall_score(y_true = y, y_pred = yhat, average=self._average, zero_division=0.0)

        if return_dataframe:
            return DataFrame({"precision": score_precision,
                              "recall":    score_recall,
                              "f1":        score_f1}, index=[0])

        return f"Precision: {score_precision:.2%}\nRecall: {score_recall:.2%}\nF1: {score_f1:.2%}"