import torch
from typing import Optional, Any

from .embedder import Embedder
from ..tracker import Tracker
from ..cache import Cache
from ..domain import UseCase, Document
from ..context import Context, ContextArg
from ..tokenizer.tokenizer import Tokenizer
from ..tracker import Tracker
from ..vectorizer.vectorizer import Vectorizer
from ..traits import Trackable, Cacheable
from ..capabilities import CudaCapable, InferenceCapable, ParameterCapable


class DerivedEmbedder(Embedder, CudaCapable, InferenceCapable, ParameterCapable, Trackable):
    r"""
    Derived embedder is a construct that allows the combination of separate instances of a tokenizer
    and vectorizer into one object.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 vectorizer: Vectorizer):

        super().__init__()

        self._tracker = Tracker()
        self._tokenizer  = tokenizer
        self._vectorizer = vectorizer

        if isinstance(vectorizer, Cacheable):
            self._cache = vectorizer.cache
            self._use_cache = vectorizer.is_caching
        else:
            self._cache = Cache(max_entries = 1024)

        self._use_cache = False
        self._is_trained = False

    def register_usecases(self, usecases: list[UseCase]):
        r"""
        Registeres a usecase with the tokenizer and the vectorizer

        :param usecases: list of relevant usecases

        :return: None
        """

        self._tokenizer.register_usecases(usecases)
        self._vectorizer.register_usecases(usecases)

    def train(self,
              ctx: Optional[Context] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              stopwords: list[str] = None,
              **kwargs) -> None:
        r"""
        Updates the vocabs of the tokenizer and the number of embeddings of the vectorizer

        Vectorizer weights are not optimized further!

        :param ctx: optional Context
        :param documents: list of documents
        :param stopwords: optional list of stopwords
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if self._is_trained:
            return

        self._tokenizer.train(ctx, documents=documents, stopwords=stopwords, **kwargs)
        vocab_size = len(self._tokenizer.get_token_set())
        self._vectorizer.train(ctx, vocab_size=vocab_size, **kwargs)
        self._is_trained = True

    def transform(self,
                  ctx: Optional[Context] = None,
                  document: Document | ContextArg[Document] = None,
                  documents: list[Document] | ContextArg[list[Document]] = None,
                  **kwargs) -> dict[str, Any]:

        r"""
        Transforms a document or documents to a tensor of word embeddings

        :param ctx: optional Context
        :param document: optional document
        :param documents: optional documents
        :param kwargs: optional, not specified list of keyword variables

        :return: dictionary {"embedder": self, "embeddings": embeddings}
        """

        tout = self._tokenizer.transform(ctx, document=document, documents=documents, **kwargs)
        tkwargs = {**kwargs, **tout}
        vout = self._vectorizer.transform(ctx, **tkwargs)

        out = {k: v for k, v in {**tout, **vout}.items() if k not in ["tokenizer", "vectorizer"]}
        out["embedder"] = self

        return out

    @property
    def embedding_dim(self) -> int:
        return self._vectorizer.embedding_dim

    def embed(self, text: str | list[str] | tuple[str, str]) -> torch.Tensor:

        if not self._use_cache:
            tokens     = self._tokenizer.tokenize(text=text)
            embeddings = self._vectorizer.vectorize(tokens=tokens)

            return embeddings

        if not isinstance(text, list):
            text = [text]

        all_embeddings = []
        for t in text:

            emb = self._cache.get(t)
            if emb is None:
                self._tracker.log(f"[CM] Embedder cache miss for '{t}'")

                tokens = self._tokenizer.tokenize(text=t)
                emb = self._vectorizer.vectorize(tokens=tokens).squeeze(0)
                self._cache.add(t, emb)
            else:
                self._tracker.log(f"[CH] Embedder cache hit for '{t}'")

            emb = self.to_active_device(emb)
            all_embeddings.append(emb)

        embeddings = torch.stack(all_embeddings, dim=0)

        return embeddings

    def set_cuda_mode(self, on: bool = True):
        r"""
        Sets cuda mode for the vectorizer on or off

        :param on: transfers the vectorizer onto the GPU if True, else the CPU

        :return: None
        """
        super().set_cuda_mode(on)

        if isinstance(self._vectorizer, CudaCapable):
            self._vectorizer.set_cuda_mode(on)

    def get_cuda_components(self) -> list[Any]:
        r"""
        Returns vectorizer's cuda-capable components

        :return: list of cuda-capable components
        """
        if isinstance(self._vectorizer, CudaCapable):
            return self._vectorizer.get_cuda_components()

        return []

    def set_inference_mode(self, on: bool, **kwargs):
        r"""
        Sets vectorizer's inference mode on or off

        :param on: turns vectorizer's inference mode if True
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        super().set_inference_mode(on)

        if isinstance(self._vectorizer, InferenceCapable):
            self._vectorizer.set_inference_mode(on)

    def get_inferential_components(self) -> list[Any]:
        r"""
        Returns vectorizer's inference-capable components

        :return: list of inference-capable components
        """

        if isinstance(self._vectorizer, InferenceCapable):
            return self._vectorizer.get_inferential_components()

        return []

    def get_training_parameters(self) -> list[Any]:
        r"""
        Returns vectorizer's training parameters

        :return: list of training parameters
        """

        if isinstance(self._vectorizer, ParameterCapable):
            return self._vectorizer.get_training_parameters()

        return []

    def configure_cache(self, on: bool, max_entries: int = 1024):
        self._cache.configure_cache(on, max_entries)
        self._use_cache = on

        self._tokenizer.configure_cache(on=on, max_entries=max_entries)
        self._vectorizer.configure_cache(on=on, max_entries=max_entries)

    def clear_cache(self):
        self._cache.clear_cache()

        self._tokenizer.clear_cache()
        self._vectorizer.clear_cache()

    @property
    def cache(self) -> Cache:
        return self._cache

    @property
    def is_caching(self) -> bool:
        return self._use_cache

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker

        if isinstance(self._tokenizer, Trackable):
            self._tokenizer.use_progress_tracker(tracker)

        if isinstance(self._vectorizer, Trackable):
            self._vectorizer.use_progress_tracker(tracker)

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    def clean(self) -> None:
        self.clear_cache()

        self._tokenizer.clean()
        self._vectorizer.clean()