from typing import Any
from pandas import DataFrame

from ...domain import UseCase, Document
from ...prediction import RawPrediction
from .metric import Metric


class Accuracy(Metric):
    r"""
    Calculates global and per-label accuracy defined as: number of matches divided by number of true labels

    Compatible with binary classifiers
    """

    @property
    def name(self) -> str:
        return "accuracy"

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:

        y    = [label for d in documents for label in usecase.get_document_label_ids(d)]
        yhat = [entry.label_id for rp in raw_predictions for entry in rp]
        labels = usecase.get_usecase_labels()

        assert len(y) > 0, "No labels available"
        assert len(y) == len(yhat), "True label count and prediction count don't match"

        accuracy = {"global": 0, "per_label": {label: 0 for label in labels}}
        totals = {label: 0 for label in labels}
        for real, pred in zip(y, yhat):
            accuracy["global"] += int(real==pred)/len(y)
            accuracy["per_label"][labels[pred]] += int(real==pred)
            totals[labels[pred]] +=1

        for k, v in totals.items():
            accuracy["per_label"][k] /= max(v, 1)

        if return_dataframe:
            df = DataFrame({"label": accuracy["per_label"].keys(),
                            "accuracy": accuracy["per_label"].values(),
                            "overall": accuracy["global"]}) \
                .sort_values("label") \
                .reset_index(drop = True)

            return df

        return accuracy