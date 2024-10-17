import torch
from abc import ABCMeta, abstractmethod
from typing import Optional

from ..context import Context
from ..traits import Trainable, Predictable
from ..domain import Document
from ..losses.loss_strategy import LossStrategy


class SupervisedModel(Trainable, Predictable, metaclass = ABCMeta):

    @abstractmethod
    def use_loss_functions(self, loss_functions: list[LossStrategy]):
        r"""
        Tells the model to use provided loss functions

        :param loss_functions: list of loss functions / strategies
        :return:
        """

        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_losses(self, ctx: Optional[Context], documents: list[Document]) -> list[torch.Tensor]:
        r"""
        Calculates loss values with all the loss functions

        :param ctx: optional Context
        :param documents: list of documents

        :return: list of loss tensors
        """

        raise NotImplementedError("interface method not implemented")