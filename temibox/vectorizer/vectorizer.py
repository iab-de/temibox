from abc import ABCMeta, abstractmethod
from typing import Any

from ..traits import Trainable, Transformable, Cacheable, Cleanable


class Vectorizer(Trainable, Transformable, Cacheable, Cleanable, metaclass=ABCMeta):
    r"""Vectorizer interface

    Although not strictly necessary, this Vectorizer abstract
    class extends the NeuralModule class. The reason for this
    is to be able to use the implemented utility methods inside
    NeuralModule. Many downstream Predictable steps work with
    PyTorch Tensors that *have* to be located on the same device
    (CPU, GPU) as the models. Extending NeuralModule gives the
    Vectorizer access to all the required utility methods without
    duplicating them here.

    """

    @abstractmethod
    def vectorize(self, tokens: Any) -> Any:
        r"""
        Vectorizes a tokenized document and returns a vector

        While the inputs are not strictly specified, a vectorizer
        should be able to ingest one or more tokenized documents
        and produce corresponding document vectors

        :param tokens: vectorizer compatible tokens

        :return: any vector representation of the document
        """

        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        r"""
        Returns the vectorizers dimension

        Many downstream tasks must know the size of the document
        vector coming out of the vectorizer

        :return: number of dimensions in vectorizers output
        """

        raise NotImplementedError("interface method not implemented")