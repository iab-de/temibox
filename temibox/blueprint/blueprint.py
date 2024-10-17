from abc import ABCMeta, abstractmethod
from typing import Type, Optional

from ..pipeline.pipeline import Pipeline
from ..prediction import Prediction
from ..traits import Trainable, Predictable, Evaluating, Trackable
from ..context import Context
from ..domain import UseCase, Document


class Blueprint(Trainable, Predictable, Evaluating, Trackable, metaclass=ABCMeta):

    @property
    @abstractmethod
    def pipeline(self) -> Pipeline:
        r"""
        Returns the pipeline instance

        :return: active pipeline instance
        """
        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def usecase(self) -> UseCase:
        r"""
        Returns the usecase instance

        :return: active usecase instance
        """
        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def document_type(self) -> Type[Document]:
        r"""
        Returns the class of the blueprint's document.
        This class can be used to instantiate new documents of the type

        :return: blueprint-specific Document class
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def train(self,
              ctx: Optional[Context] = None,
              documents: list[Document] = None,
              **kwargs) -> None:
        r"""
        Runs the pipeline training workflow (pipeline.train)

        :param ctx: optional Context
        :param documents: optional list of documents
        :param kwargs: other parameters possibly relevant to pipeline steps (see blueprint.pipeline.get_signature(Trainable))

        :return: None
        """

        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def predict(self,
                ctx: Optional[Context] = None,
                document: Document | list[Document] = None,
                **kwargs) -> list[Prediction]:
        r"""
        Runs the pipeline prediction workflow (pipeline.predict)

        :param ctx: optional Context
        :param document: optional document or list of documents
        :param kwargs: other parameters possibly relevant to pipeline steps (see blueprint.pipeline.get_signature(Predictable))

        :return: list of predictions (one per document)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def evaluate(self,
                 ctx: Optional[Context] = None,
                 documents: list[Document] = None,
                 **kwargs) -> list[str]:
        r"""
        Runs the pipeline evaluation workflow (pipeline.evaluate)

        :param ctx: optional Context
        :param documents: optional list of documents
        :param kwargs: other parameters possibly relevant to pipeline steps (see blueprint.pipeline.get_signature(Evaluating))

        :return: list of strings (one metric per string)
        """
        raise NotImplementedError("interface method not implemented")