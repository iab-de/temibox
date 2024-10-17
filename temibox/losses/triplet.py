import torch
import torch.nn as nn

from typing import Any

from ..capabilities import CudaCapable
from .loss_strategy import LossStrategy
from ..domain import UseCase, Document, Label


class TripletLoss(LossStrategy):
    r"""
    Calculates triplet loss for a given set of documents.

    For each document a number of triplets are generated from the usecase
    data.
    """

    def __init__(self,
                 use_class_weights: bool,
                 examples: int = 1,
                 scale: float = 1.0):

        self._use_weights = use_class_weights
        self._examples = examples
        self._scale = scale

    def __call__(self,
                 embedder: Any,
                 usecase: UseCase,
                 documents: list[Document],
                 **kwargs) -> torch.Tensor:

        triplets = [t for d in documents for t in usecase.create_document_triplet(d, self._examples)]

        if not len(triplets):
            loss = torch.zeros(1)
            if isinstance(embedder, CudaCapable):
                loss = embedder.to_active_device(loss)
            return loss

        positive = embedder.embed(text = [t.positive for t in triplets])
        anchor   = embedder.embed(text = [t.anchor   for t in triplets])
        negative = embedder.embed(text = [t.negative for t in triplets])

        if self._use_weights:
           d_weights  = torch.FloatTensor([w for d in documents for w in self._examples * [sum(usecase.get_label_inverse_weights(usecase.get_document_labels(d)))]]).unsqueeze(0)

           if isinstance(embedder, CudaCapable):
               d_weights = embedder.to_active_device(d_weights)

           criteria   = nn.TripletMarginLoss(reduction="none")
           raw_losses = criteria(anchor, positive, negative).unsqueeze(0)
           loss       = torch.mm(d_weights, raw_losses.T).squeeze(1)/sum(d_weights.squeeze(0))
        else:
           criteria = nn.TripletMarginLoss(reduction="none")
           loss = (criteria(anchor, positive, negative).sum()/len(documents)).unsqueeze(0)

        return loss * self._scale