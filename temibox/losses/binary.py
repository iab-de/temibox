import torch
from typing import Any

from .loss_strategy import LossStrategy
from ..embedder.embedder import Embedder
from ..capabilities import BinaryNeuralModel
from ..domain import UseCase, Document, Label


class BinaryLoss(LossStrategy):
    r"""
    Calculates a (weighted) binary cross entropy loss for the binary case
    """

    def __init__(self,
                 use_class_weights: bool,
                 scale: float = 1.0):

        self._use_weights = use_class_weights
        self._scale = scale

    def __call__(self,
                 model: BinaryNeuralModel,
                 usecase: UseCase,
                 documents: list[Document],
                 embedder: Embedder = None,
                 **kwargs) -> torch.Tensor:

        texts:       list[str]         = [usecase.get_document_body(d) for d in documents]
        labels:      list[list[Label]] = [usecase.get_document_labels(d) for d in documents]
        label_ids:   list[list[int]]   = [usecase.get_document_label_ids(d) for d in documents]
        labels_flat: list[Label]       = [label for d in labels for label in d]

        assert len(labels) == len(label_ids), "Number of labels and label IDs do not match"
        assert len(texts) == len(label_ids), "Label count does not match document count"
        assert len(texts) == len(labels_flat), "Label count does not match document count (flattened)"

        document_embeddings = embedder.embed(text=texts)
        document_embeddings = model.to_active_device(document_embeddings)

        if self._use_weights:
            weights  = torch.FloatTensor(usecase.get_label_inverse_weights(labels_flat)).unsqueeze(1)
            weights  = model.to_active_device(weights)
            criteria = torch.nn.BCELoss(weight=weights)
        else:
            criteria = torch.nn.BCELoss()

        y    = model.to_active_device(torch.FloatTensor(label_ids))
        yhat = model.forward(document_embeddings)

        return criteria(yhat, y) * self._scale
