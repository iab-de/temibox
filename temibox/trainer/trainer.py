from abc import ABCMeta
from ..traits import Trainable


class Trainer(Trainable, metaclass=ABCMeta):
    r"""
    Placeholder type representing a model trainer
    """
    pass