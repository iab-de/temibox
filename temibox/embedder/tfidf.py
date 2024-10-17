import torch
import locale
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Set, Any, Dict, Optional

from ..tokenizer.tokenizer import Tokenizer
from ..tokenizer.ngram import NgramTokenizer
from ..capabilities import CudaCapable
from ..context import  Context, ContextArg
from ..domain import Document
from ..cache import Cache
from .embedder import Embedder


class TFIDFEmbedder(Embedder, CudaCapable):

    def __init__(self,
                 max_tokens: int,
                 embedding_dim: int,
                 min_token_frequency: int = 5,
                 tokenizer: Tokenizer = None):
        r"""
        TF-IDF embedder

        :param max_tokens:         max number of tokens
        :param embedding_dim:      embedding (SVD) dimension
        :param tokenizer:          tokenizer
        """

        super().__init__()

        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = NgramTokenizer(min_token_frequency = min_token_frequency,
                                             max_tokens          = max_tokens,
                                             cased               = False)

        self._tokens        = []
        self._targets       = []
        self._token_pos     = {}
        self._idf           = {}
        self._max_tfidf     = -1
        self._svd_matrix    = None
        self._max_tokens    = max_tokens
        self._embedding_dim = embedding_dim
        self._cuda          = False
        self._cuda_components = []

        self._use_cache = False
        self._cache = Cache(max_entries=1024)

    def _get_tf(self, abs_tf, max_f) -> float:
        return 0.5 + 0.5 * abs_tf / max_f

        ###################################
        # Embedder methods
        ###################################

    def get_training_parameters(self) -> List[Any]:
        return []

    def train(self,
              ctx: Optional[Context] = None,
              documents: List[Document] | ContextArg[list[Document]] = None,
              stopwords: Set[str] = None,
              **kwargs) -> None:

        r"""
        Calculates TF-IDF values and performs single value decomposition of the resulting matrix

        :param ctx: optional Context
        :param documents: list of documents
        :param stopwords: optional list of stopwords
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        self._tokenizer.train(**{"ctx": ctx, "documents": documents, "stopwords": stopwords or set(), **kwargs})

        documents = ContextArg.extract(documents)

        self._targets = sorted(list({c for uc in ctx.usecases for d in documents for c in uc.get_document_labels(d)}), key=locale.strxfrm)
        self._tokens = sorted(list(self._tokenizer.get_token_set()), key=locale.strxfrm)
        self._token_pos = {t: i for i, t in enumerate(self._tokens)}

        abs_df = {}
        abs_tf = {}
        for doc in tqdm(documents, "Learning Embeddings"):
            for uc in ctx.usecases:
                abs_df_seen = set()
                targets = uc.get_document_labels(doc)
                for token in self._tokenizer.tokenize(text=uc.get_document_body(doc))[0]:

                    if token not in abs_df_seen:
                        abs_df[token] = abs_df.get(token, 0) + 1
                        abs_df_seen.add(token)

                    for target in targets:
                        if target not in abs_tf:
                            abs_tf[target] = {}

                        abs_tf[target][token] = abs_tf[target].get(token, 0) + 1

        for uc in ctx.usecases:
            for target in uc.get_usecase_labels():
                abs_df_seen = set()
                for token in self._tokenizer.tokenize(text=target)[0]:

                    if token not in abs_df_seen:
                        abs_df[token] = abs_df.get(token, 0) + 1
                        abs_df_seen.add(token)

                    if target not in abs_tf:
                        abs_tf[target] = {}

                    abs_tf[target][token] = abs_tf[target].get(token, 0) + 1

        # IDF
        self._idf = {}
        for i, (token, count) in enumerate(abs_df.items()):
            count = min(count, len(documents))
            self._idf[token] = np.log((1 + len(documents) - count) / (1 + count))

        # TF
        tf = {}
        for target, tokens in abs_tf.items():
            max_f = max([v for v in tokens.values()])
            tf[target] = {}
            for token in tokens:
                tf[target][token] = self._get_tf(abs_tf[target][token], max_f)

        # TF-IDF Tensor
        self._max_tokens = min(len(self._token_pos), self._max_tokens)
        tf_idf_tensor = torch.FloatTensor(self._max_tokens, len(self._targets))
        for j, target in tqdm(enumerate(self._targets), "Creating TF-IDF matrix"):
            for token in tf[target]:
                tf_idf_tensor[self._token_pos[token]][j] = tf[target].get(token, 0) * self._idf.get(token, 0)

        tf_idf_tensor = tf_idf_tensor.nan_to_num()

        self._max_tfidf = tf_idf_tensor.max().item()
        tf_idf_tensor /= self._max_tfidf

        # SVD
        svd_U, _, _ = torch.svd(tf_idf_tensor)

        if self._embedding_dim > svd_U.shape[1]:
            self._embedding_dim = svd_U.shape[1]

        if self._embedding_dim < tf_idf_tensor.shape[0]:
            svd_U = svd_U[:, :self._embedding_dim]

        self._cuda_components.append(svd_U)

        self._svd_matrix = torch.mm(tf_idf_tensor.T, svd_U)
        self._svd_matrix /= self._svd_matrix.norm(dim=1)[:, None]

    def transform(self,
                  ctx: Optional[Context] = None,
                  text: str | List[str] | Tuple[str, str] = None,
                  document: Any = None,
                  documents: List[Any] = None,
                  **kwargs) -> Dict[str, Any]:

        r"""
        Transforms a document or documents to a tensor of embeddings

        The embeddings are a result of the TF-IDF values of the documents
        scaled using SVD to achieve the required embedding size

        :param ctx: optional Context
        :param text: optional raw text
        :param document: optional document
        :param documents: optional documents
        :param kwargs: optional, not specified list of keyword variables

        :return: dictionary {"embedder": self, "embeddings": embeddings}
        """

        results = {"embedder": self}

        if text:
            results["embeddings"] = self.embed(text=text)
        elif document:
            results["embeddings"] = self.embed(text=ctx.active_usecase.get_document_body(document))
        elif documents and not ctx.pipeline.is_training:
            results["embeddings"] = self.embed(text=[ctx.active_usecase.get_document_body(d) for d in documents])

        return results

    def clean(self) -> None:
        self.clear_cache()

    ###################################
    # Embedder methods
    ###################################

    def get_cuda_components(self) -> List[Any]:
        return self._cuda_components

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, text: str | List[str] | Tuple[str, str]) -> torch.Tensor:

        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, tuple):
            texts = list(text)
        else:
            texts = text

        svd_results = torch.zeros((len(texts), self._embedding_dim))

        for k, text in enumerate(texts):

            text_tokens = self._tokenizer.tokenize(text=text)

            if (svd_result := self.cache.get(text)) is None:

                result = torch.zeros((len(text_tokens), self._max_tokens), dtype=torch.float)

                for i, tokens in enumerate(text_tokens):
                    abs_tf = Counter(tokens)
                    max_f = abs_tf.most_common(1)[0][1] if len(abs_tf.most_common(1)) else 1
                    tf = {k: self._get_tf(v, max_f) for k, v in abs_tf.items()}

                    for token in tokens:
                        if j := self._token_pos.get(token, None):
                            result[i][j] = tf.get(token, 0) * self._idf.get(token, 0)

                result /= self._max_tfidf
                result = self.to_active_device(result)

                svd_result = torch.mm(result, self._cuda_components[0])
                svd_result /= svd_result.norm(dim=1)[:, None]

                if svd_result.isnan().any().item():
                    svd_result = torch.zeros_like(svd_result)

                if self.is_caching:
                    self.cache.add(text, svd_result)

            svd_results[k,:] = svd_result

        return self.to_active_device(svd_results)

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