from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from .tracker import Tracker
from .prediction import Prediction
from .context import Context
from .domain import UseCase


class PipelineStep(metaclass=ABCMeta):
    r"""Main trait describing a pipeline step

    This trait should not be implemented directly,
    but only extended by other pipeline step traits
    describing actual pipeline actions (see Trainable)
    """

    def __init__(self):
        self._registered_usecases: dict[str, UseCase] = {}

    @property
    def registered_usecases(self) -> list[UseCase]:
        r"""
        Returns the step's usecases

        :return: list of usecases
        """
        return list(self._registered_usecases.values())

    @property
    def registered_usecase_names(self) -> list[str]:
        r"""
        Returns the list of step's usecase names (not instances!)

        :return: sorted list of usecase names
        """
        return sorted(list(self._registered_usecases.keys()))

    def register_usecases(self, usecases: list[UseCase]) -> None:
        r"""
        Registers a list of usecases with the step.
        The step will consider these and only these usecases as relevant.
        Should not contain usecases with the same name (UseCase.name)

        :param usecases: list of usecases

        :return: None
        """

        if not hasattr(self, "_registered_usecases"):
            self._registered_usecases = {}

        for usecase in usecases:
            self._registered_usecases[usecase.name] = usecase

class Trainable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a trainable pipeline step

    Training can be seen as a *necessary* preparation
    for the other pipeline steps (transformation, prediction,
    etc.). What this preparation entails must be decided by the
    implementing class. In some cases (e.g. econometric models),
    the training step results in an estimated statistical model,
    while in other cases, it could be as little as storing a
    certain data structure inside the step to be used later.
    """

    @abstractmethod
    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        r"""
        Runs step's training workflow

        :param ctx: optional Context
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """
        raise NotImplementedError("trait method not implemented")

class Transformable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a transforming pipeline step

    Transformation is any kind of action that returns
    a modification if its inputs or some new input that
    might be needed by steps along the pipeline.

    In many cases a Transformable step is also a Trainable step.
    """

    @abstractmethod
    def transform(self, ctx: Optional[Context] = None, **kwargs) -> dict[str, Any]:
        r"""
        Runs step's transformation workflow

        :param ctx: optional Context
        :param kwargs: optional, not specified list of keyword variables

        :return: dict[str, Any] - products of the transformation
        """
        raise NotImplementedError("trait method not implemented")


class Cleanable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a cleanable pipeline step

    Steps creating or storing transient data, especially large
    amounts (training documents, etc.) should implement this
    trait, so that he pipeline can clean up before exporting
    """

    @abstractmethod
    def clean(self) -> None:
        r"""
        Clean up the step

        What clean does must be defined by the concrete implementation
        of the step trait. Possible uses: clear data caches,
        remove references to large object in order to avoid them
        being serialized when exporting the pipeline and so on.

        :return: None
        """
        raise NotImplementedError("trait method not implemented")


class Cacheable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a cacheable pipeline step

    A pipeline step might profit from caching its results.
    This trait represents a step with a configurable fixed size
    cache
    """

    @abstractmethod
    def configure_cache(self, on: bool, max_entries: int = 1024) -> None:
        r"""
        Configures the cache

        :param on: turn the cache on (True) or off (False)
        :param max_entries: max number of entries in the cache (default = 1024)

        :return: None
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def clear_cache(self) -> None:
        r"""
        Removes all the entries from the cache

        :return: None
        """
        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def cache(self) -> 'Cache':
        r"""
        Returns the cache object used by the step

        :return: its own cache object
        """

        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def is_caching(self) -> bool:
        r"""
        Returns whether the step is caching or not

        :return: True if cache is active
        """
        raise NotImplementedError("interface method not implemented")

class Predictable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a predicting pipeline step

    Usually a prediction is just a transformation if some
    data (input -> prediction). However, in order to distinguish
    predictions from transformations, so that the step's results could
    be more easily consumed by the pipeline's users, a Predictable
    step produces a different type of output - a list of instances
    of the Prediction type
    """

    @abstractmethod
    def use_identifier(self, identifier: str):
        r"""
        Gives the predictable step a custom identifier

        :param identifier: string identifier

        :return: None
        """

        raise NotImplementedError("trait method not implemented")

    @abstractmethod
    def predict(self, ctx: Optional[Context], **kwargs) -> list[Prediction]:
        r"""
        Returns a list of predictions

        :param ctx: optional Context
        :param kwargs: an optional, not specified list of keyword variables
        :return: list[Prediction] - list of predictions
        """
        raise NotImplementedError("trait method not implemented")


class Trackable(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing a trackable pipeline step

    The pipeline step initializes an empty tracker that
    *can* be used by the step. To actually be useful,
    either an extension of the Tracker class, or the basic
    Tracker instance with an injected progress function
    should be used.
    """

    @abstractmethod
    def use_progress_tracker(self, tracker: Tracker) -> None:
        r"""Injects a tracker

        The tracker *can* be used to inform the pipeline or
        external entities about the progress being made
        by the step. The uses of the tracker / progress
        notification *must* be manually defined in the
        concrete implementation of the step trait

        :param tracker: an instance of a Tracker

        :return: None
        """
        raise NotImplementedError("trait method not implemented")

    @abstractmethod
    def get_progress_tracker(self) -> Tracker:
        r"""Returns step's tracker

        :return: an instance of a Tracker
        """
        raise NotImplementedError("trait method not implemented")


class Evaluating(PipelineStep, metaclass=ABCMeta):
    r"""Trait describing an evaluating pipeline step

    Evaluating steps are used in pipeline performance evaluation
    The output of an evaluation is step specific and might involve
    both inputs (e.g. documents to be evaluated) or outputs (e.g.
    results of the performance metrics)
    """

    @abstractmethod
    def evaluate(self, ctx: Optional[Context] = None, **kwargs) -> dict[str, Any]:
        r"""
        Evaluates the input

        :param ctx: optional Context
        :param kwargs: an optional, not specified list of keyword variables

        :return: dictionary of evaluation-relevant inputs or outputs
        """
        raise NotImplementedError("trait method not implemented")

class Supervising(PipelineStep, metaclass=ABCMeta):
    r"""Supervises other steps

    Intended to be used by supporting steps, e.g. trainers
    of neural networks, in order to be able to select relevant
    steps to train
    """

    @abstractmethod
    def supervise(self, name: str, step: PipelineStep, **kwargs) -> None:
        r"""Tells the supervising step to supervise this child step

        :param name: name of the child step
        :param step: child step to  be supervised by this step

        :return: None
        """
        raise NotImplementedError("trait method not implemented")

    @abstractmethod
    def supervising(self) -> list[str]:
        raise NotImplementedError("trait method not implemented")