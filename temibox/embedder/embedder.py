import torch
from abc import ABCMeta, abstractmethod
from typing import Any

from ..traits import Trainable, Transformable, Cacheable, Cleanable


class Embedder(Trainable, Transformable, Cacheable, Cleanable, metaclass=ABCMeta):
    r"""Text embedder (transforms strings into two dimensional tensors)

    """

    @abstractmethod
    def get_training_parameters(self) -> list[Any]:
        r"""
        Returns training parameters (if any)

        :return: list of trainable parameters
        """

        raise NotImplementedError("abstract method not implemented")

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        r"""
        Returns the embedding dimension

        :return: embedding dimension
        """

        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def embed(self, text: str | list[str] | tuple[str, str]) -> torch.Tensor:
        r"""
        Tokenizes and vectorizes the provided text and returns an embedding.
        The shape of the embedding depends on the number of targets provided
        and the vectorizer used.

        :param text: a string, list of strings or tuple of two strings
                     that should be embedded.
                     In case of the tuple, the second string is to be used
                     as additional context

        :return: a torch Tensor containing word embeddings
        """

        raise NotImplementedError("interface method not implemented")