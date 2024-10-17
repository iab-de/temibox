import torch

from .loss_strategy import LossStrategy
from ..domain import Label, Document, UseCase
from ..embedder.embedder import Embedder
from ..capabilities import MultilabelBinaryNeuralModel


class MultilabelBinaryLoss(LossStrategy):
    r"""
    Calculates a (weighted) binary cross entropy loss

    For each document a number of random test labels (positive and negative examples)
    as well as corresponding predictions are generated and a binary cross entropy calculated
    """
    def __init__(self,
                 positive_examples: int,
                 negative_examples: int,
                 use_class_weights: bool = True,
                 scale: float = 1.0):

        self._positive_examples = positive_examples
        self._negative_examples = negative_examples
        self._use_class_weights = use_class_weights
        self._scale = scale

    def __call__(self,
                 model:     MultilabelBinaryNeuralModel,
                 usecase:   UseCase,
                 documents: list[Document],
                 embedder:  Embedder = None,
                 **kwargs) -> torch.Tensor:

        test_labels = [usecase.create_document_test_labels(document = d,
                                                           positive=self._positive_examples,
                                                           negative=self._negative_examples) for d in documents]

        test_labels_str = [[x[0] for x in d] for d in test_labels]
        test_labels_idx = [[int(x[1]) for x in d] for d in test_labels]

        texts: list[str] = [usecase.get_document_body(d) for d in documents]

        document_embeddings = embedder.embed(text=texts)
        document_embeddings = model.to_active_device(document_embeddings)

        target_embeddings = torch.concat([embedder.embed(text=d_labels).unsqueeze(0) for d_labels in test_labels_str])
        target_embeddings = model.to_active_device(target_embeddings)

        if self._use_class_weights:
            binary_inv_weights = torch.concat([torch.FloatTensor(usecase.get_label_inverse_weights(ts)).unsqueeze(0) for ts in test_labels_str])
            binary_inv_weights = model.to_active_device(binary_inv_weights)
            criteria = torch.nn.BCELoss(weight=binary_inv_weights)
        else:
            criteria = torch.nn.BCELoss()

        y = torch.concat([torch.FloatTensor(ti).unsqueeze(0) for ti in test_labels_idx])
        y = model.to_active_device(y)
        yhat = model.forward(document_embeddings, target_embeddings)

        return self._scale * criteria(yhat, y)
