import numpy as np
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from typing import Type, Union


Document = Type["Document"]
Label    = Union[int, str]
Triplet  = namedtuple("Triplet", ["positive", "anchor", "negative"])

@dataclass
class LabelDescription:
    r"""
    Dataclass used to represent a simple label.

    weight should contain reasonable values (e.g. absolute frequencies) if
    inverse class weights are to be used in training (to correct for class imbalance)
    """

    label_id: int
    label: str
    weight: float = 1.0

class UseCase(metaclass=ABCMeta):

    @property
    @abstractmethod
    def name(self) -> str:
        r"""
        Usecase name (must be unique among usecases used in a single pipeline)

        :return: string name
        """
        raise NotImplementedError("interface method not implemented")

    # Core methods
    @abstractmethod
    def get_document_body(self, document: Document) -> str:
        r"""
        Returns the body of the document.

        What exactly is the body and what it contains depends on the usecase
        and the basis for the document, e.g.: in topic classification of
        scientific papers, the title and the abstract could be concatenated
        to represent the "body". However, if the publications also contain some
        keywords, then the keywords could also be included into the body.
        What *cannot* be included into the body are the topics, i.e. the labels
        that the model actually has to predict in the given usecase.

        In a keyword generation usecase, the topics might as well be included
        into the body, if it is expected that the topics are available before
        the keyword generation takes part.

        :param document: a usecase specific document

        :return: a string representing the body of the document
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_document_labels(self, document: Document) -> list[Label]:
        r"""
        Returns list of document labels (list with one element in non-multilabel cases)

        :param document: a usecase specific document

        :return: list of labels
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_document_label_ids(self, document: Document) -> list[int]:
        r""" Returns integer IDs for document classes

        :param document: a usecase specific document

        :return: list of integers
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_usecase_labels(self) -> list[Label]:
        r""" Returns all the labels available in the use case. For binary
        classification problems it's a list with two values [target A, target B],
        for a multiclass usecase it is a list with n > 2 strings

        NB: the order of the labels is important and should remain constant

        :return: list of labels
        """
        raise NotImplementedError("interface method not implemented")

    def get_usecase_label_dict(self) -> dict[int, Label]:
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_usecase_label_weights(self) -> dict[Label, float]:
        r"""
        Returns label weights that can be used for class balancing and
        prediction sorting.

        Label weights can be especially important in usecases with
        severely unbalanced classes.

        :return: dictionary of label: float values
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def create_document_test_labels(self, document: Document, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        r"""
        Returns positive+negative number of random labels
        It might be necessary to draw labels with replacement to make the required number of values

        Depending on the usecase, the way the negative examples are drawn can be more complex than
        just randomly drawing from a set of negative examples (obviously negative examples might lead
        to a slower learning and weaker generalization power of the model)

        :param document: a usecase specific document
        :param positive: number of positive examples (i.e. labels actually belonging to the document)
        :param negative: number of negative examples (labels not belonging to the document)

        :return: a list of tuples containing the label and a boolean indicator of whether the label is positive
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def create_document_triplet(self, document: Document, examples: int = 1) -> list[Triplet]:
        r"""
        Returns a triplet containing a positive example, an anchor example
        (e.g. the document body) and a negative example.

        A triplet is very similar to a call to get_binary_test_labels(document, 1, 1)
        in that it returns a single positive and a single negative example. However,
        generally a great care should be taken in drawing the negative example. The
        models using triplet loss profit the most from negative examples that are *hard*
        to classify, i.e. are not obviously wrong.
        In topic classification such an example could be a topic that the document does not have,
        but that contains several (non-stop)words that are also found in the positive topics, e.g.
        - existing positive label: The effect of unemployment benefits
        - good negative example (hard to classify): The effects of unemployment on the economy
        - bad negative example (easy to classify): New developments in international trade

        :param document: a usecase specific document
        :param examples: number of triplets to return

        :return:
        """
        raise NotImplementedError("interface method not implemented")

    # Support methods
    def get_usecase_label_inverse_weights(self) -> dict[Label, float]:
        r"""
        Returns inverse label weights that can be used for class rebalancing
        (penalizing majority classes)

        :return: list of floats
        """
        lweights = self.get_usecase_label_weights()
        weights = lweights.values()
        minw = min(weights)
        normalized_inverse_weights = [minw/(w if w > 0 else 1) for w in weights]

        return {k: normalized_inverse_weights[i] for i, k in enumerate(lweights)}

    def get_label_weights(self, labels: list[Label]) -> list[float]:
        r"""
        Returns list of label weights for the provided list of labels

        :param labels: list of labels

        :return: list of floats
        """
        weights = self.get_usecase_label_weights()

        return [weights.get(label, 0.0) for label in labels]

    def get_label_inverse_weights(self, labels: list[Label]) -> list[float]:
        r"""
        Returns a list of inverse label weights for the provided list of labels

        :param labels: list of labels

        :return: list of floats
        """

        weights = self.get_usecase_label_weights()
        minw = min(weights.values())
        normalized_inverse_weights = [minw/weights.get(label, 1.0) for label in labels]

        return normalized_inverse_weights

    def get_multinomial_indicators(self, document: Document) -> list[int]:
        r"""
        Returns a list of integers (0/1) indicating whether a label at position i is
        relevant to the document (1) or not (0). The length of the list corresponds
        to the size of the label set, i.e. each label in the set is represented by a
        value.

        N.B: the implicit order of the labels must be consistent (i.e. label at position i is
        always the same label for all i)

        :param document: a usecase specific document

        :return: list of integers
        """

        all_targets     = list(self.get_usecase_label_dict().keys())
        pos_all_targets = self.get_document_label_ids(document)

        multinomial_indicators = np.zeros(len(all_targets))
        multinomial_indicators[np.where(np.in1d(all_targets, pos_all_targets))] = 1

        return list(multinomial_indicators)


