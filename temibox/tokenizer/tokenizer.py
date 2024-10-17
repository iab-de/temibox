from abc import ABCMeta, abstractmethod
from typing import List, Set, Any

from ..traits import Trainable, Transformable, Cacheable, Cleanable


class Tokenizer(Trainable, Transformable, Cacheable, Cleanable, metaclass=ABCMeta):
    r"""Text tokenizer interface

    The tokenizer's task is to parse a piece of text and generate
    token representations. These representations can be anything
    from byte-level or character-level strings, up to ngrams and
    so on.

    """

    @abstractmethod
    def tokenize(self, text: str | List[str]) -> Any:
        r"""
        Tokenizes text and returns a tokenized document

        The tokenizer should be able to ingest anything from a raw string
        all the way to an instance of a Document and return a tokenized
        Document.

        Since a normal string is a possible input, the tokenizer should
        be able to create instances of the Document class (i.e. a List of
        Sentence instances)

        :param text:  a string, sentence, list of sentences or a Document
                    that should be tokenized.

        :return:  the output of the tokenizer is usually tokenizer-specific
        """

        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_token_set(self) -> Set[str]:
        r"""
        Returns a set of possible tokens

        The tokenizer should have a finite set of tokens, which is
        usually a necessary requirement by other downstream tasks
        (e.g. vectorization)

        :return: a set of all possible tokens
        """

        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_token_ids(self, **kwargs) -> List[int]:
        r"""
        Returns a list of token ids

        Each token must have a unique numerical ID. Some vectorizer
        might have a use for such a list in their training phase.

        :param kwargs: an optional, not specified list of keyword variables

        :return: a list of all token IDs
        """

        raise NotImplementedError("interface method not implemented")

