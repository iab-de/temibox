import spacy
from tqdm import tqdm
from typing import Dict, Any, List, Set, Tuple, Optional

from .tokenizer import Tokenizer
from ..context import Context, ContextArg
from ..traits import Trainable, Transformable
from ..cache import Cache
from ..domain import Document

_nlp = spacy.load('de_core_news_lg')


class NgramTokenizer(Tokenizer, Trainable, Transformable):


    def __init__(self,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_word_length: int = 2,
                 min_token_frequency: int = 5,
                 max_tokens: int = 30000,
                 remove_stopwords: bool = True,
                 cased: bool = False):

        super().__init__()

        self._ngram_range = ngram_range
        self._remove_stopwords = remove_stopwords
        self._min_word_length = min_word_length
        self._min_token_frequency = min_token_frequency
        self._max_tokens = max_tokens
        self._cased = cased

        self._use_cache = False
        self._cache = Cache(max_entries=1024)

        self._stopwords = set()

        self._token_dict: Dict[str, int] = {}
        self._token_set: Set[str] = set()

    ###################################
    # Pipeline methods
    ###################################

    def _get_token_key(self, ngram: str | List[str]) -> str:

        if isinstance(ngram, str):
            return "_".join(ngram.replace("_","-").split()).lower()

        return "_".join([w.replace("_", "-").lower() for w in ngram])

    def _create_tokens(self,
                       text: str,
                       return_known_only: bool = False) -> List[str]:

        words = text.split()

        tokens = []
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(0, len(words) - n + 1):
                ngram = words[i:i + n]

                # Remove stopwords
                if self._remove_stopwords and len(self._stopwords) and any([w in self._stopwords for w in ngram]):
                    continue

                # Remove short words
                if any([len(w) < self._min_word_length for w in ngram]):
                    continue

                token = " ".join(ngram)
                if not self._cased:
                    token = token.lower()

                token_key = self._get_token_key(ngram)

                if token_key in self._token_dict:
                    tokens.append(self._token_dict[token_key])
                elif not return_known_only:
                    tokens.append(token)

        return tokens

    def train(self,
              ctx: Optional[Context] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              stopwords: Set[str] = None,
              **kwargs) -> None:

        r"""
        Trains the tokenizer

        :param ctx: optional Context
        :param documents: list of documents
        :param stopwords: optional set of stopwords
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if stopwords:
            self._stopwords = stopwords.copy()

        documents = ContextArg.extract(documents)

        token_counter = {}
        for doc in tqdm(documents, "Tokenizing"):

            for uc in ctx.usecases:

                # Classes as tokens
                for s in uc.get_document_labels(doc):
                    stokens = self._create_tokens(str(s).strip(".!?;:,"), return_known_only=False)

                    for token in stokens:
                        token_counter[token] = token_counter.get(token, 0) + 1

                # Text as tokens
                for s in _nlp(uc.get_document_body(doc)).sents:
                    lemmas = " ".join([w.lemma_ for w in s if w.lemma_ != "--"])
                    stokens = self._create_tokens(lemmas, return_known_only=False)

                    for token in stokens:
                        token_counter[token] = token_counter.get(token, 0) + 1

        # Take _max_tokens top tokens
        token_counter = [(k, v) for k, v in token_counter.items() if v >= self._min_token_frequency]
        token_counter = sorted(token_counter, key=lambda x: x[1])[::-1][:self._max_tokens]

        # Token set
        self._token_dict = {self._get_token_key(v): v for (v, _) in token_counter}
        self._token_set = set(self._token_dict.values())

    def transform(self,
                  ctx: Optional[Context] = None,
                  text: str = None,
                  document: Document | ContextArg[Document] = None,
                  documents: list[Document] | ContextArg[list[Document]] = None,
                  **kwargs) -> Dict[str, Any]:
        r"""
        Extracts tokens from text

        either the raw text, a document or a list of documents have to be provided

        :param ctx: optional Context
        :param text: raw text
        :param document: a single document
        :param documents: a list of documents

        :param kwargs: optional, not specified list of keyword variables

        :return: dictionary containing tokens
        """

        if text:
            textlist = [text]
        elif document:
            textlist = [ctx.active_usecase.get_document_body(ContextArg.extract(document, ctx.active_usecase.name))]
        elif documents:
            textlist = [ctx.active_usecase.get_document_body(d) for d in ContextArg.extract(documents, ctx.active_usecase.name)]
        else:
            return {}

        tokens = []
        for text in textlist:
            i_tokens = []

            for t in [str(s).strip(".") for s in _nlp(text).sents]:

                if (toks := self._cache.get(t)) is None:
                   toks = self._create_tokens(t, return_known_only=True)

                i_tokens += toks

            tokens.append(i_tokens)

        return {"tokens": tokens}

    ###################################
    # Tokenizer methods
    ###################################

    def get_token_set(self) -> Set[str]:
        return self._token_set

    def get_token_ids(self, text: str, **kwargs) -> List[Any]:
        tokens = self.tokenize(text=text, return_tokens=True)

        return [self._get_token_key(t) for t in tokens]

    def tokenize(self, text: str | List[str] | Tuple[str, str], **kwargs) -> List[str]:

        tokens = []
        if isinstance(text, str):
            tokens += self.transform(text=text).get("tokens", [])

        elif isinstance(text, list):
            tokens += [ti for t in text for ti in self.transform(text=t).get("tokens", [])]

        elif isinstance(text, Tuple) and len(text) == 2:
            if text[0]:
                tokens += self.transform(text=text[0]).get("tokens", [])
            if text[1]:
                tokens += self.transform(text=text[1]).get("tokens", [])

        return tokens

    ###################################
    # Cache methods
    ###################################

    def configure_cache(self, on: bool, max_entries: int = 1024):
        self._cache.configure_cache(on, max_entries)
        self._use_cache = on

    def clear_cache(self):
        self._cache.clear_cache()

    @property
    def cache(self) -> Cache:
        return self._cache

    @property
    def is_caching(self) -> bool:
        return self._use_cache

    def clean(self) -> None:
        self.clear_cache()

