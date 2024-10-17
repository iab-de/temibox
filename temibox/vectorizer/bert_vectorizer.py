import torch
from typing import Optional, Any, List
from transformers import AutoModel

from .vectorizer import Vectorizer
from ..tracker import Tracker
from ..cache import Cache
from ..traits import Trackable
from ..capabilities import CudaCapable, InferenceCapable, ParameterCapable
from ..context import Context, ContextArg


class BertVectorizer(Vectorizer, CudaCapable, InferenceCapable, ParameterCapable, Trackable):

    def __init__(self, pretrained_model_dir: str):

        super().__init__()

        self._use_cache = False
        self._cache = Cache(max_entries = 1024)
        self._vectorizer = AutoModel.from_pretrained(pretrained_model_dir, local_files_only=True)
        self._tracker = Tracker()
        self._models = [self._vectorizer]

        self._embedding_dim = self._vectorizer.get_input_embeddings().embedding_dim
        self._is_trained = False

    def train(self,
              ctx: Optional[Context] = None,
              vocab_size: int | ContextArg[int] = None,
              **kwargs) -> None:

        if self._is_trained:
            return

        if ctx is None:
            raise Exception("No context provided")

        if vocab_size is None:
            raise Exception("Unknown vocab size")

        if isinstance(vocab_size, ContextArg):
            vocab_size = max([vocab_size.get(uname) for uname in self.registered_usecase_names]) or 0

        if vocab_size > self._vectorizer.config.vocab_size:
            self._vectorizer.resize_token_embeddings(vocab_size)

        self._is_trained = True

    def transform(self,
                  ctx: Optional[Context] = None,
                  texts: list[str] | ContextArg[list[str]] = None,
                  tokens: dict[str, torch.Tensor] | ContextArg[dict[str, torch.Tensor]] = None,
                  **kwargs) -> dict[str, Any]:

        if ctx is None or ctx.active_usecase is None:
            raise Exception("No context or active usecase provided")

        if ctx.active_usecase.name not in self.registered_usecase_names:
            return {}

        results = {"vectorizer": self}

        # Embeddings not needed in training mode
        if ctx.pipeline.is_training:
            return results

        tokens = ContextArg.extract(tokens, ctx.active_usecase.name)

        if self._use_cache and texts is not None:
            texts = ContextArg.extract(texts, ctx.active_usecase.name)

            if len(texts) == tokens.get("input_ids").shape[0]:
                all_embeddings = []
                for i, t in enumerate(texts):
                    emb = self._cache.get(t)
                    if emb is None:
                        self._tracker.log(f"[CM] Vectorizer cache miss for '{t}'")

                        emb = self.vectorize(tokens = {"input_ids": tokens["input_ids"][[i],:],
                                                       "attention_mask": tokens["attention_mask"][[i],:]}).squeeze(0)
                        self._cache.add(t, emb)
                    else:
                        self._tracker.log(f"[CH] Vectorizer cache hit for '{t}'")

                    all_embeddings.append(emb)

                embeddings = torch.stack(all_embeddings, dim=0)
        elif tokens is None:
            raise Exception("No tokens provided")
        else:
            embeddings = self.vectorize(tokens=tokens)

        results["embeddings"] = embeddings

        return results

    def vectorize(self,
                  tokens: dict[str, torch.Tensor],
                  return_mean_embedding: bool = True,
                  **kwargs) -> torch.Tensor:

        ctokens = {k: self.to_active_device(v) for k,v in tokens.items()}
        if return_mean_embedding:
            embeddings = self._vectorizer(**ctokens).last_hidden_state.mean(dim=1)
        else:
            embeddings = self._vectorizer(**ctokens).last_hidden_state

        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_cuda_components(self) -> List[Any]:
        return self._models.copy()

    def get_inferential_components(self) -> List[Any]:
        return self._models.copy()

    def get_training_parameters(self) -> Any:
        return [p for m in self._models for p in m.parameters() if p.requires_grad]

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

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    def clean(self) -> None:
        self.clear_cache()