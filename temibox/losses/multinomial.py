import torch
from typing import Any


from .loss_strategy import LossStrategy
from ..capabilities import MultinomialNeuralModel
from ..embedder.embedder import Embedder
from ..domain import UseCase, Document


class MultinomialLoss(LossStrategy):
    r"""
    Calculates a (weighted) binary cross entropy loss for the multinomial case
    """

    def __init__(self,
                 use_class_weights: bool = True,
                 scale: float = 1.0,
                 positive_weight: float = 10.0):

        self._use_class_weights = use_class_weights
        self._positive_weight = positive_weight
        self._scale = scale

    def __call__(self,
                 model: MultinomialNeuralModel,
                 usecase: UseCase,
                 documents: list[Document],
                 embedder: Embedder = None,
                 **kwargs) -> torch.Tensor:

        texts: list[str] = [usecase.get_document_body(d) for d in documents]
        document_embeddings = embedder.embed(text=texts)
        document_embeddings = model.to_active_device(document_embeddings)

        y = torch.concat([torch.FloatTensor(usecase.get_multinomial_indicators(d)).unsqueeze(0) for d in documents])
        y = model.to_active_device(y)

        yhat = model.forward(usecase_name=usecase.name,
                             document_embeddings=document_embeddings)

        if self._use_class_weights:
            binary_inv_weights = torch.FloatTensor(list(usecase.get_usecase_label_inverse_weights().values())).repeat(len(documents), 1)
            binary_inv_weights = model.to_active_device(binary_inv_weights)
            criteria = torch.nn.BCELoss(weight=binary_inv_weights)
        else:
            if self._positive_weight > 1:
                pos_w = min(self._positive_weight, len(usecase.get_usecase_labels()))
                criteria = torch.nn.BCELoss(weight = y * pos_w + 1)
            else:
                criteria = torch.nn.BCELoss()

        return self._scale * criteria(yhat, y)