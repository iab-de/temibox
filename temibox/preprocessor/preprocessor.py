from abc import ABCMeta, abstractmethod
from typing import List, Any

from ..traits import Trainable, Transformable


class Preprocessor(Trainable, Transformable, metaclass=ABCMeta):
    r"""Text preprocessor interface
    """

    @abstractmethod
    def process(self, text: str | List[str]) -> str | List[str] | Any:
        r"""Prepares text and converts it into a Document

        Args:
            text - input string

        Returns:
            cleaned up text as an instance of Document
        """
        raise NotImplementedError("interface method not implemented")


