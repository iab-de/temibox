from abc import ABCMeta
from ...traits import Trainable, Transformable, Predictable


class Summarizer(Trainable, Transformable, Predictable, metaclass=ABCMeta):
    r"""
    Placeholder type representing a summarizer
    """
    pass