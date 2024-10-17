from typing import Optional, Any

import torch

from temibox.context import Context
from temibox.model.classifier import BinaryClassifier
from temibox.capabilities import CudaCapable, InferenceCapable
from temibox.interfaces import Embedder

class MockEmbedder(Embedder):

    def get_training_parameters(self) -> list[Any]:
        pass

    @property
    def embedding_dim(self) -> int:
        return 8

    def embed(self, text: str | list[str] | tuple[str, str]) -> torch.Tensor:
        pass

    def train(self, ctx: Optional[Context] = None, **kwargs) -> None:
        pass

    def transform(self, ctx: Optional[Context] = None, **kwargs) -> dict[str, Any]:
        pass

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


def test_inference():

    model = BinaryClassifier(multilabel = False, layer_dimensions = [8])

    assert isinstance(model, InferenceCapable), "Binary classifier should be inference capable"
    assert isinstance(model, CudaCapable), "Binary classifier should be cuda capable"
    assert not model.is_inference, "Created model should be in training mode"
    assert not model.is_cuda, "Created model should not be loaded onto the GPU"
    assert len(model.get_training_parameters()) == 0, "Untrained model has no parameters"

    model.set_inference_mode(on=True)
    model.set_inference_mode(on=False)
    model.train(embedder = MockEmbedder())
    assert len(model.get_training_parameters()) > 0, "Trained model has parameters"

    model.set_inference_mode(on = True)
    assert model.is_inference, "Model should be in inference mode"
    assert len(model.get_training_parameters()) == 0, "Model in inference mode has no trainable parameters"

    model.zero_grad()
    model.set_inference_mode(on=False)
    model.zero_grad()


def test_cuda():

    model = BinaryClassifier(multilabel = False, layer_dimensions = [8])
    model.train(embedder=MockEmbedder())

    assert not model.is_cuda, "Created model should not be loaded onto the GPU"
    model.set_cuda_mode(on = True)
    assert model.is_cuda, "Model should be loaded onto the GPU (1)"
    assert model._model.parameters().__next__().device.type == "cuda", "Model should be loaded onto the GPU (2)"

    x = torch.FloatTensor([1,2,3])
    assert x.device.type == "cpu"
    x = model.to_active_device(x)
    assert x.device.type == "cuda"

    model.set_cuda_mode(on=False)
    assert not model.is_cuda, "Model should be loaded onto the CPU"

    model2 = BinaryClassifier(multilabel=False, layer_dimensions=[8])
    model2.set_cuda_mode(on=True)
    assert model2.is_cuda, "Model2 should be loaded onto the GPU"
    model2.set_cuda_mode(on=False)
    assert not model2.is_cuda, "Model2 should be loaded onto the CPU"


def test_cuda_before_train():

    model = BinaryClassifier(multilabel=False, layer_dimensions=[8])
    assert not model.is_cuda, "Created model should not be loaded onto the GPU"
    model.set_cuda_mode(on=True)
    assert model.is_cuda, "Model should be loaded onto the GPU (1)"
    model.train(embedder=MockEmbedder())
    assert model._model.parameters().__next__().device.type == "cuda", "Model should be loaded onto the GPU (2)"