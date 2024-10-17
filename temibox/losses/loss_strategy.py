import torch
from typing import Any
from abc import ABCMeta, abstractmethod

from ..domain import Document, UseCase


class LossStrategy(metaclass=ABCMeta):
    r"""
    Represents a loss function.

    It is called a loss strategy, because the loss function is allowed to do more
    than just calculate the loss, e.g. create random samples, make predictions, etc.
    """

    @abstractmethod
    def __call__(self,
                 model:     Any,
                 usecase:   UseCase,
                 documents: list[Document],
                 **kwargs) -> torch.Tensor:

        r"""
        Calculates the loss value within the provided context (usecase) for a list of documents and a given model

        :param model: predictable model
        :param usecase:  active usecase
        :param documents: list of usecase-specific documents
        :param kwargs: optional, not specified list of keyword variables

        :return: torch tensor
        """

        raise NotImplementedError("interface method not implemented")
