import numpy as np
from pandas import DataFrame
from typing import Any

from .metric import Metric
from ...domain import UseCase, Document
from ...prediction import RawPrediction


class ConfusionMatrix(Metric):
    r"""
    Calculates classification confusion matrix

    Compatible with binary and multinomial / multicategorical classifiers
    """
    def __init__(self,
                 max_predictions: int = 10,
                 min_score:       float = 0.5,
                 show_percent:    bool = False):

        self._show_percent = show_percent
        self._max_predictions = max_predictions
        self._min_score = min_score

    @property
    def name(self) -> str:
        return f"confusion matrix{' (%)' if self._show_percent else ''}"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 **kwargs) -> Any:

        assert len(documents) == len(raw_predictions), "Prediction count and document count don't match"

        label_ids = set(usecase.get_usecase_label_dict().keys())
        cm = np.zeros((len(label_ids), 4), dtype=int if not self._show_percent else float)

        for i, doc in enumerate(documents):
            t_labels_doc = set(usecase.get_document_label_ids(doc))
            rel_predictions = max(self._max_predictions, len(t_labels_doc))
            p_labels_doc = {p.label_id for p in sorted(raw_predictions[i], key=lambda x: x.score, reverse=True)[:rel_predictions] if p.score >= self._min_score}

            for j, label in enumerate(label_ids):
                cm[j,0] += label in     t_labels_doc and label in     p_labels_doc
                cm[j,1] += label not in t_labels_doc and label in     p_labels_doc
                cm[j,2] += label in     t_labels_doc and label not in p_labels_doc
                cm[j,3] += label not in t_labels_doc and label not in p_labels_doc

        df_pred = DataFrame(cm)
        df_pred.columns = ["TP", "FP", "FN", "TN"]
        df_pred.index = list(usecase.get_usecase_label_dict().values())
        df_pred["label_id"] = list(usecase.get_usecase_label_dict().keys())
        df_pred["relevant"] = (df_pred.TP+df_pred.FP+df_pred.FN) > 0
        df_pred["precision"] = df_pred.apply(lambda x: x.TP / (x.TP+x.FP) if (x.TP+x.FP) > 0 else 0.0, axis=1)
        df_pred["recall"] = df_pred.apply(lambda x: x.TP / (x.TP + x.FN) if (x.TP + x.FN) > 0 else 0.0, axis=1)
        df_pred["f1"] = df_pred.apply(lambda x: 2*(x.precision*x.recall)/(x.precision+x.recall) if (x.precision+x.recall) > 0 else 0.0 , axis=1)

        if self._show_percent:
            df_pred["_total"] = df_pred.apply(lambda x: x.TP + x.FP + x.FN + x.TN, axis=1)
            df_pred["TP"] = df_pred.apply(lambda x: f"{x.TP / x._total:.2%}", axis=1)
            df_pred["TN"] = df_pred.apply(lambda x: f"{x.TN / x._total:.2%}", axis=1)
            df_pred["FP"] = df_pred.apply(lambda x: f"{x.FP / x._total:.2%}", axis=1)
            df_pred["FN"] = df_pred.apply(lambda x: f"{x.FN / x._total:.2%}", axis=1)
            df_pred = df_pred.drop(columns = ["_total"])

        return df_pred[["label_id", "relevant", "TP", "FP", "FN", "TN", "precision", "recall", "f1"]]