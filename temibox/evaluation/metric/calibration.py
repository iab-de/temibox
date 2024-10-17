import numpy as np
from pandas import DataFrame
from typing import Any

from temibox.domain import UseCase, Document
from temibox.prediction import RawPrediction
from temibox.evaluation.metric.metric import Metric

class Calibration(Metric):
    r"""
    Calculates actual/predicted label calibration

    Compatible with binary classifiers
    """

    @property
    def name(self) -> str:
        return "calibration"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 **kwargs) -> Any:

        y = [label for d in documents for label in usecase.get_document_label_ids(d)]
        yhat = [entry.label_id for rp in raw_predictions for entry in rp]

        assert len(y) > 0, "No labels available"
        assert len(y) == len(yhat), "True label count and prediction count don't match"

        labels = usecase.get_usecase_labels()
        ca = np.zeros((len(labels), 2), dtype=float)

        for i in range(len(y)):
            ca[y[i], 0]    += 1/len(y)
            ca[yhat[i], 1] += 1/len(y)

        df_pred = DataFrame(ca)

        for i in range(len(labels)):
            df_pred.loc[i, 0] = f"{df_pred.loc[i, 0]:.2%}"
            df_pred.loc[i, 1] = f"{df_pred.loc[i, 1]:.2%}"

        df_pred.index = [f"{label}" for label in labels]
        df_pred.columns = ["real", "pred"]

        return df_pred