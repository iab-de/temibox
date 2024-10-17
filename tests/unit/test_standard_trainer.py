import pytest
import torch
from typing import Optional, Any, List
from dataclasses import dataclass

from temibox.context import Context
from temibox.prediction import Prediction
from temibox.pipeline import StandardPipeline
from temibox.trainer import StandardTrainer
from temibox.traits import Trainable, Predictable
from temibox.capabilities import ParameterCapable, InferenceCapable
from temibox.interfaces import SupervisedModel, Embedder
from temibox.domain import Label, UseCase, Triplet, Document
from temibox.losses import LossStrategy


#######################
# Mocks
#######################
@dataclass
class MockDocument:
    text: str
    label_id: int

class MockUseCase(UseCase):

    def __init__(self, name: str = "test-usecase"):
        self._name = name
        self._labels = ["a", "b"]

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

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return {"a": 0.5, "b": 0.5}

    def create_document_test_labels(self, document: MockDocument, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        return [("a", document.label_id == 0),
                ("b", document.label_id == 1)]

    def create_document_triplet(self, document: MockDocument, examples: int = 1) -> list[Triplet]:
        return []

class MockEmbedder(Embedder, InferenceCapable):

    def get_inferential_components(self) -> List[Any]:
        return []

    def get_training_parameters(self) -> list[Any]:
        return []

    @property
    def embedding_dim(self) -> int:
        pass

    def embed(self, text: str | list[str] | tuple[str, str]) -> torch.Tensor:
        pass

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        pass

    def transform(self, ctx: Optional[Context] = None, **kwargs) -> dict[str, Any]:
        return {"embeddings": []}

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


class MockModel(SupervisedModel, ParameterCapable, InferenceCapable):

    def __init__(self, loss_functions: list[LossStrategy]):
        super().__init__()

        self._model = None
        self._loss_functions = loss_functions
        self._identifier = "mock-model"

    def use_identifier(self, identifier: str):
        self._identifier = identifier

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        self._model = torch.nn.Sequential(*[torch.nn.Linear(1,1), torch.nn.Sigmoid()])

    def predict(self, ctx: Optional[Context], **kwargs) -> list[Prediction]:
        pass

    def get_training_parameters(self) -> Any:
        return self._model.parameters()

    def use_loss_functions(self, loss_functions: list[LossStrategy]):
        self._loss_functions = loss_functions

    def forward(self, doc_count, **kwargs):
        return self._model(torch.rand(doc_count).unsqueeze(1))

    def get_losses(self, ctx: Optional[Context], documents: list[Document]) -> list[torch.Tensor]:

        losses = []
        for fn in self._loss_functions:
            losses += fn(model     = self,
                         usecase   = ctx.active_usecase,
                         documents = documents)

        return losses

    def get_inferential_components(self) -> List[Any]:
        return [self._model]


class MockLoss(LossStrategy):

    def __call__(self,
                 model: MockModel,
                 usecase: UseCase,
                 documents: list[Document],
                 **kwargs) -> torch.Tensor:

        return model.forward(len(documents))


#######################
# Setup
#######################

Trainplan = StandardTrainer.Trainplan

trainplans = [Trainplan(epochs=1, learn_rate=1e-3, batch_size=1, freeze_vectorizer=True),
              Trainplan(epochs=1, learn_rate=1e-4, batch_size=1, freeze_vectorizer=False, max_docs=10),
              Trainplan(epochs=2, learn_rate=1e-5, batch_size=32, freeze_vectorizer=False)]

documents = [MockDocument(text="Das ist ein Test", label_id=0),
             MockDocument(text="Zweites Testdokument", label_id=1),
             MockDocument(text="Das letzte Testdokument", label_id=0)]

#######################
# Tests
#######################
def test_train():

    trainer = StandardTrainer()
    embedder = MockEmbedder()
    model = MockModel(loss_functions=[MockLoss()])

    pipeline = StandardPipeline() \
        .add_usecase(MockUseCase()) \
        .add_step("model", model) \
        .add_step("embedder", embedder) \
        .add_step("trainer", trainer, dependencies=["embedder", "model"])

    with pytest.raises(Exception):
        pipeline.train(trainplans = trainplans,
                       documents = None)

    pipeline.train(trainplans=trainplans,
                   documents=documents.copy() * 20)

    pipeline.train(trainplans = None,
                   documents=documents.copy())

    assert len(trainer._loss_history) > 0, "Loss history exists"
    trainer.clean()
    assert len(trainer._loss_history) == 0, "Loss history is clean exists"

def test_plot_history():
    trainer = StandardTrainer()
    model = MockModel(loss_functions=[MockLoss()])

    pipeline = StandardPipeline() \
        .add_usecase(MockUseCase()) \
        .add_step("model", model) \
        .add_step("trainer", trainer, dependencies=["model"])

    pipeline.train(trainplans=trainplans,
                   documents=documents.copy() * 20)

    trainer.plot_history()

def test_empty_plot_history():
    trainer = StandardTrainer()

    assert trainer.plot_history() is None, "Should not throw an exception"