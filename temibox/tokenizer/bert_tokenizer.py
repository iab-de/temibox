import re
import torch
import logging
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Optional
from transformers import AutoTokenizer
from collections import Counter

from .tokenizer import Tokenizer
from ..cache import Cache
from ..domain import Document
from ..context import Context, ContextArg


class BertTokenizer(Tokenizer):
    r"""
    Tokenizer used by the BERT vectorizer
    """

    _logger = logging.getLogger("BertTokenizer")

    def __init__(self,
                 pretrained_model_dir: str,
                 allow_max_new_tokens: int = 256,
                 min_new_token_freq: int = 32,
                 max_sequence_length: int = 512):

        super().__init__()

        self._allow_max_new_tokens = allow_max_new_tokens
        self._min_new_token_freq = min_new_token_freq
        self._max_sequence_length = max_sequence_length
        self._added_tokens = 0
        self._use_cache = False
        self._cache = Cache(max_entries = 1024)
        self._is_trained = False
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                        local_files_only=True,
                                                        do_lower_case=False)

    def train(self,
              ctx: Optional[Context] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              stopwords: list[str] = None,              
              **kwargs) -> None:

        r"""
        Trains the tokenizer

        :param ctx: optional Context
        :param documents: list of documents
        :param stopwords: optional set of stopwords
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if self._is_trained:
            return

        if self._allow_max_new_tokens <= 0:
            return

        if ctx is None:
            raise Exception("No context provided")

        texts = []
        for usecase in ctx.usecases:
            if usecase.name not in self.registered_usecase_names:
                continue

            docs = ContextArg.extract(documents, usecase.name)

            if docs is None:
                raise Exception("No documents provided")

            texts += [usecase.get_document_body(d) for d in docs]
        
        stopwords = stopwords or set()

        counter = Counter()
        for text in tqdm(texts, "Learning new tokens"):
            words = re.sub(r"[^a-zäöüßÄÖÜß0-9 ]+", " ", text, flags=re.I).split()
            clean_words = [w for w in words if len(w.strip()) >= 2 and w.lower() not in stopwords]

            counter.update(clean_words)

        additional_tokens = [t for (t, c) in counter.most_common(self._allow_max_new_tokens) if c >= self._min_new_token_freq]

        if additional_tokens:
            added = self._tokenizer.add_tokens(additional_tokens,
                                               special_tokens=False)  # BUG. siehe https://github.com/huggingface/tokenizers/issues/507
        else:
            added = 0

        self._added_tokens = added

        self._logger.info(f"Added {added} new tokens")
        self._is_trained = True

    def transform(self,
                  ctx: Optional[Context] = None,
                  document: Document | ContextArg[Document] = None,
                  documents: list[Document] | ContextArg[list[Document]]  = None,
                  **kwargs) -> dict[str, Any]:
        r"""
        Extracts tokens from text

        either the document or a list of documents have to be provided

        :param ctx: optional Context
        :param document: a single document
        :param documents: a list of documents

        :param kwargs: optional, not specified list of keyword variables

        :return: dictionary containing raw texts and tokens
        """

        if ctx is None or ctx.active_usecase is None:
            raise Exception("No context provided")

        if ctx.active_usecase.name not in self.registered_usecase_names:
            return {}

        results = {"tokenizer": self,
                   "vocab_size": self._tokenizer.vocab_size}

        # Tokens not needed in training mode
        if ctx.pipeline is not None and ctx.pipeline.is_training:
            return results

        # Get tokens
        if document is not None and (doc := ContextArg.extract(document, ctx.active_usecase.name)):
            texts = [ctx.active_usecase.get_document_body(doc)]

        elif documents is not None and (docs := ContextArg.extract(documents, ctx.active_usecase.name)):
            texts = [ctx.active_usecase.get_document_body(d) for d in docs]

        else:
            raise Exception("No document(s) provided")

        results["texts"]  = texts
        results["tokens"] = self.tokenize(texts)

        return results

    def _get_tokens(self, text: str | list[str] | tuple[str, str]) -> dict[str, torch.Tensor]:
        if isinstance(text, tuple):
            t1, t2 = text
        else:
            t1, t2 = text, None

        return self._tokenizer(text=t1,
                               text_pair=t2,
                               truncation=True,
                               padding=True,
                               return_tensors="pt",
                               max_length=self._max_sequence_length,
                               is_split_into_words=False)

    def tokenize(self, text: str | list[str] | tuple[str, str]) -> dict[str, torch.Tensor]:

        if not self._use_cache:
            return self._get_tokens(text)

        if not isinstance(text, list):
            text = [text]

        token_list = []
        maxtoks = 0
        for t in text:

            if (tokens := self._cache.get(t)) is None:
                tokens = self._get_tokens(t)
                self._cache.add(t, tokens)

            token_list.append(tokens)
            if (tlen := tokens["input_ids"].shape[1]) > maxtoks:
                maxtoks = tlen

        # Combine
        pad_id = self._tokenizer.convert_tokens_to_ids("[PAD]")
        combo = {"input_ids": [F.pad(tok["input_ids"], (pad_id, maxtoks-tok["input_ids"].shape[1])).squeeze(0) for tok in token_list],
                 "attention_mask": [F.pad(tok["attention_mask"], (0, maxtoks-tok["attention_mask"].shape[1])).squeeze(0) for tok in token_list]}

        combo["input_ids"]      = torch.stack(combo["input_ids"], dim=0)
        combo["attention_mask"] = torch.stack(combo["attention_mask"], dim=0)

        return combo

    def get_token_set(self) -> set[str]:
        return set(self._tokenizer.get_vocab().keys())

    def get_token_ids(self, **kwargs) -> list[int]:
        return list(self._tokenizer.get_vocab().values())

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

