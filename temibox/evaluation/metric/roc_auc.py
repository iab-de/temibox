from sklearn.metrics import roc_auc_score
from typing import Any
from pandas import DataFrame

from .metric import Metric
from ...domain import UseCase, Document
from ...prediction import RawPrediction


class RocAuc(Metric):
    r"""
    Calculates ROC-AUC

    Compatible with binary classifiers
    """
    @property
    def name(self) -> str:
        return "roc-auc"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:

        y_true  = [label for d in documents for label in usecase.get_document_label_ids(d)]
        y_score = [pi.score for p in raw_predictions for pi in p]
        rauc    = roc_auc_score(y_true=y_true, y_score=y_score)

        if return_dataframe:
            return DataFrame({"ROC-AUC": rauc}, index=[0])

        return f"ROC-AUC: {rauc:.2%}"