import torch
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Set, Optional, Any

from temibox.model.classifier import BinaryClassifier
from temibox.interfaces import Tokenizer, Vectorizer, Embedder, Metric
from temibox.losses import LossStrategy
from temibox.domain import UseCase, Document, Label, Triplet
from temibox.context import Context, ContextArg
from temibox.traits import PipelineStep, Supervising, Evaluating, Trainable, Transformable, Predictable, Cleanable, Trackable, Cacheable
from temibox.capabilities import CudaCapable, InferenceCapable
from temibox.cache import Cache
from temibox.prediction import Prediction, RawPrediction
from temibox.tracker import Tracker


@dataclass
class MockDocument:
    text: str
    label_id: int

@dataclass
class MockDocumentMulti:
    text: str
    label_ids: list[int]

class MockLoss(LossStrategy):

    def __call__(self,
                 model: BinaryClassifier,
                 usecase: UseCase,
                 documents: list[Document],
                 **kwargs) -> torch.Tensor:

        return torch.FloatTensor([len(documents)])

class MockEmbedder(Embedder):

    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._last_embeddings = []

    def get_training_parameters(self) -> list[Any]:
        return []

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, text: str | list[str] | tuple[str, str]) -> torch.Tensor:
        num = 1 if isinstance(text, str) else len(text)
        self._last_embeddings.append(torch.rand((num, self._embedding_dim)))

        return self._last_embeddings[-1]

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        pass

    def transform(self,
                  ctx: Optional[Context] = None,
                  documents: list[MockDocument | MockDocumentMulti] = None,
                  **kwargs) -> dict[str, Any]:

        text = [ctx.active_usecase.get_document_body(d) for d in documents]
        embeddings = self.embed(text=text)

        return {"embedder": self,
                "embeddings": embeddings}

    def configure_cache(self, on: bool, max_entries: int = 1024):
        pass

    def clear_cache(self):
        pass

    @property
    def cache(self) -> 'Cache':
        return None

    @property
    def is_caching(self) -> bool:
        return False

    def clean(self) -> None:
        pass

class MockTokenizer(Tokenizer):

    def tokenize(self, text: str | List[str]) -> Any:
        text = text if isinstance(text, list) else [text]

        return [["token A", "token B"]] * len(text)

    def get_token_set(self) -> Set[str]:
        pass

    def get_token_ids(self, **kwargs) -> List[int]:
        pass

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        pass

    def transform(self,
                  ctx: Optional[Context] = None,
                  documents: list[MockDocument] = None,
                  **kwargs) -> dict[str, Any]:

        if documents is None:
            return {}

        return {"tokens": ["token A", "token B"] * len(documents)}

    def configure_cache(self, on: bool, max_entries: int = 1024):
        pass

    def clear_cache(self):
        pass

    @property
    def cache(self) -> 'Cache':
        pass

    @property
    def is_caching(self) -> bool:
        pass

    def clean(self) -> None:
        pass

class MockVectorizer(Vectorizer):

    def __init__(self, embedding_dim: int = 16):
        super().__init__()

        self._embedding_dim = embedding_dim

    def vectorize(self, tokens: Any) -> Any:
        return torch.rand((len(tokens), self._embedding_dim))

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        pass

    def transform(self,
                  ctx: Optional[Context] = None,
                  tokens: list[list[any]] = None,
                  **kwargs) -> dict[str, Any]:

        return torch.rand((len(tokens), self._embedding_dim))

    def configure_cache(self, on: bool, max_entries: int = 1024):
        pass

    def clear_cache(self):
        pass

    @property
    def cache(self) -> 'Cache':
        pass

    @property
    def is_caching(self) -> bool:
        pass

    def clean(self) -> None:
        pass

class MockUseCase(UseCase):

    def __init__(self, name: str = "test-usecase", no_triplets: bool = False):
        self._name = name
        self._labels = ["a", "b"]
        self._weights = {"a": 0.75, "b": 0.25}
        self._label_dict = {i:label for i, label in enumerate(self._labels)}
        self._no_triplets = no_triplets

    @property
    def name(self) -> str:
        return self._name

    def get_document_body(self, document: MockDocument) -> str:
        return document.text

    def get_document_labels(self, document: MockDocument) -> list[Label]:
        return [self._labels[document.label_id]]

    def get_document_label_ids(self, document: MockDocument) -> list[int]:
        return [document.label_id]

    def get_usecase_labels(self) -> list[Label]:
        return self._labels.copy()

    def get_usecase_label_dict(self) -> dict[int, Label]:
        return self._label_dict

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return self._weights

    def create_document_test_labels(self, document: MockDocument, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        return [(self._labels[document.label_id], True)] * positive + [(self._labels[1-document.label_id], False)]  * negative

    def create_document_triplet(self, document: MockDocument, examples: int = 1) -> list[Triplet]:
        if self._no_triplets:
            return []

        ok_label = self._labels[document.label_id]
        nok_label = self._labels[1-document.label_id]
        triplet = Triplet(ok_label, document.text, nok_label)

        return [triplet]*examples


class MockMultiUseCase(MockUseCase):

    def __init__(self, name: str = "test-usecase", use_labels: dict[str, tuple[int, float]] = None):
        super().__init__()

        self._name = name
        if use_labels is None:
            self._labels = ["a", "b", "c"]
            self._weights = {"a": 0.25, "b": 0.25, "c": 0.5}
        else:
            self._labels = list(use_labels.keys())
            self._label_dict = {v[0]: k for k,v in use_labels.items()}
            self._weights = {k:v[1] for k,v in use_labels.items()}

    def get_document_label_ids(self, document: MockDocumentMulti) -> list[int]:
        return document.label_ids

    def get_document_labels(self, document: MockDocumentMulti) -> list[Label]:
        return [self._labels[label-1] for label in document.label_ids]


class MockClasslessUseCase(UseCase):

    @property
    def name(self) -> str:
        return "MockClasslessUseCase"

    def get_document_body(self, document: MockDocument) -> str:
        return document.text

    def get_document_labels(self, document: MockDocument) -> list[Label]:
        return []

    def get_document_label_ids(self, document: MockDocument) -> list[int]:
        return []

    def get_usecase_labels(self) -> list[Label]:
        return []

    def get_usecase_label_dict(self) -> dict[int, Label]:
        return {}

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return {}

    def create_document_test_labels(self, document: MockDocument, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        return []

    def create_document_triplet(self, document: MockDocument, examples: int = 1) -> list[Triplet]:
        return []


class TestTrainable(Trainable, Cleanable, CudaCapable, InferenceCapable, Trackable, Cacheable):

    def __init__(self,
                 name: str,
                 activities: list[Any],
                 sleep: bool=False,
                 fail_clean: bool = False):

        super().__init__()
        super(CudaCapable).__init__()
        super(InferenceCapable).__init__()

        self._name = name
        self._activities = activities
        self._tracker = Tracker()
        self._sleep = sleep
        self._fail_clean = fail_clean

        self._max_entries = 1024
        self._is_caching = False
        self._cache = Cache(max_entries=self._max_entries)

        self._inference = False
        self._cuda_mode = False

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        self._activities.append(self._name + "-train")
        if self._sleep:
            time.sleep(2)

    def clean(self) -> None:
        self._activities.append(self._name + "-clean")
        if self._fail_clean:
            raise Exception("Failed on purpose")

    def get_cuda_components(self) -> List[Any]:
        return []

    def get_inferential_components(self) -> List[Any]:
        return []

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    def configure_cache(self, on: bool, max_entries: int = 1024):
        self._activities.append(f"{self._name}-configure-cache")

        self._is_caching = on
        if not on or max_entries != self._max_entries:
            self.clear_cache()
            self._cache = Cache(max_entries=max_entries)
            self._max_entries = max_entries

    def clear_cache(self):
        self._activities.append(f"{self._name}-clear-cache")
        self._cache.clear_cache()

    @property
    def cache(self) -> 'Cache':
        return self._cache

    @property
    def is_caching(self) -> bool:
        return self._is_caching

    def set_cuda_mode(self, on: bool = True):
        self._cuda_mode = on

    def set_inference_mode(self, on: bool, **kwargs):
        self._inference = on

    @property
    def is_cuda(self):
        return self._cuda_mode

    @property
    def is_inference(self):
        return self._inference

class TestTrainableTransformable(Trainable, Transformable):

    def __init__(self, name: str, activities: list[Any]):
        super().__init__()

        self._name = name
        self._activities = activities
        self._fail = False

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        self._activities.append(self._name + "-train")

    def transform(self,
                  ctx: Optional[Context] = None,
                  initial_value: ContextArg[int] = None,
                  **kwargs) -> dict[str, Any]:

        self._activities.append(self._name + "-transform")

        if self._fail:
            raise Exception("forced to fail")

        if initial_value is None:
            return {}

        return {"transformed_value": ContextArg.extract(initial_value) * 2}

    def fail_transform(self):
        self._fail = True

class TestTransformableOnly(Transformable, Evaluating):

    def __init__(self, name: str, activities: list[Any]):
        super().__init__()

        self._name = name
        self._activities = activities

    def transform(self,
                  ctx: Optional[Context] = None,
                  initial_value: ContextArg[int] = None,
                  **kwargs) -> dict[str, Any]:

        self._activities.append(self._name + "-transform")

        if initial_value is None:
            return {}

        return {"transformed_value": ContextArg.extract(initial_value) * 2}

    def evaluate(self, ctx: Optional[Context] = None, **kwargs):
        self._activities.append(self._name + "-evaluate")
        return self.transform(ctx, **kwargs)

class TestTrainablePredictable(Trainable, Predictable):

    def __init__(self, name: str, activities: list[Any]):
        super().__init__()

        self._name = name
        self._activities = activities

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        self._activities.append(self._name + "-train")

    def use_identifier(self, identifier: str):
        pass

    def predict(self,
                ctx: Optional[Context],
                transformed_value: ContextArg[int] = None,
                reps: int = 1,
                **kwargs) -> list[Prediction]:

        self._activities.append(self._name + "-predict")
        tv = ContextArg.extract(transformed_value, ctx.active_usecase.name)

        pred = Prediction(usecase_name = ctx.active_usecase.name,
                          model        = self._name,
                          timestamp    = int(datetime.now().timestamp()),
                          payload_raw  = [RawPrediction(label="a", label_id=0, score=0.25),
                                          RawPrediction(label="b", label_id=1, score=0.50),
                                          RawPrediction(label="c", label_id=2, score=0.75)],
                          payload      = tv)

        return [pred]*reps

class TestSupervisor(Supervising):

    def __init__(self, namelist: set[str]):
        super().__init__()
        self._supervising = namelist

    def supervise(self, name: str, step: PipelineStep, **kwargs) -> None:
        self._supervising.add(name)

    def supervising(self) -> list[str]:
        return list(self._supervising)


class TestEvaluator(Evaluating):

    def __init__(self, name: str, activities: list[Any], fail: bool = False):
        super().__init__()

        self._name = name
        self._activities = activities
        self._fail = fail

    def fail_evaluations(self):
        self._fail = True

    def evaluate(self,
                 ctx: Optional[Context] = None,
                 transformed_value: ContextArg[int] = None,
                 **kwargs) -> Any:
        self._activities.append(self._name + "-evaluate")

        if self._fail:
            raise Exception("Failing on purpose")

        tv = ContextArg.extract(transformed_value, ctx.active_usecase.name)

        return {"evaluation_result": tv*3}

class TestUseCase(UseCase):

    def __init__(self, name: str = "test-usecase"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_document_body(self, document: Document) -> str:
        pass

    def get_document_labels(self, document: Document) -> list[Label]:
        pass

    def get_document_label_ids(self, document: Document) -> list[int]:
        pass

    def get_usecase_labels(self) -> list[Label]:
        pass

    def get_usecase_label_weights(self) -> dict[Label, float]:
        pass

    def get_usecase_label_dict(self) -> dict[int, Label]:
        pass

    def create_document_test_labels(self, document: Document, positive: int, negative: int) -> list[
        tuple[(Label, bool)]]:
        pass

    def create_document_triplet(self, document: Document, examples: int = 1) -> list[Triplet]:
        pass

class MockMetric(Metric):

    def __init__(self, name: str = "mock-metric"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self,
                 raw_predictions: list[list[RawPrediction]],
                 usecase: UseCase, documents: list[Document],
                 transformed_value: float = 0,
                 **kwargs) -> Any:


        return transformed_value * 3