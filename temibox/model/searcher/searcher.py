from abc import ABCMeta
from ...traits import Trainable, Transformable, Predictable


class Searcher(Trainable, Transformable, Predictable, metaclass=ABCMeta):
    r"""
    Placeholder type representing a search engine
    """
    pass