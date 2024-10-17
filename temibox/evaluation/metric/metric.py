from abc import ABCMeta, abstractmethod
from typing import Any
from ...domain import UseCase, Document
from ...prediction import RawPrediction

class Metric(metaclass=ABCMeta):
    r"""
    Metric used for pipeline performance evaluation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        r"""
        Returns metric name

        :return: metric name
        """
        pass

    @abstractmethod
    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase,
                 documents: list[Document],
                 return_dataframe: bool = False,
                 **kwargs) -> Any:
        r"""
        Calculates the metric value(s)

        :param raw_predictions: list of lists (i.e. one list per document) of raw predictions (temibox.prediction.RawPrediction)
        :param usecase: active usecase
        :param documents: list of source documents
        :param return_dataframe: returns a dataframe if True, else a string
        :param kwargs: optional, not specified list of keyword variables

        :return: string or dataframe
        """

        raise NotImplementedError("interface method not implemented")