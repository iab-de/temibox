import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from pandas import DataFrame
from collections import OrderedDict
from typing import Optional, Any, Type

from ..classifier.classifier import Classifier
from ...tokenizer.tokenizer import Tokenizer
from ...tracker import Tracker
from ...vectorizer.vectorizer import Vectorizer
from ...embedder.embedder import Embedder
from ...embedder.derived_embedder import DerivedEmbedder
from ...traits import Trackable
from ...capabilities import CudaCapable, InferenceCapable, ParameterCapable
from ...prediction import RawPrediction, Prediction
from ...domain import Label, Document
from ...losses import LossStrategy, BinaryLoss, MultilabelBinaryLoss
from ...context import Context, ContextArg


class BinaryClassifier(Classifier, CudaCapable, InferenceCapable, ParameterCapable, Trackable):
    r"""
    Binary classifier to be used as a shallow network on top of a transformer-based embedder.

    Can be used for binary and multilabel use cases
    """
    def __init__(self,
                 multilabel:        bool,
                 layer_dimensions:  list[int],
                 layer_activation:  Type[torch.nn.Module] = torch.nn.ReLU,
                 dropout:           float = 0.0,
                 loss_functions:    list[LossStrategy] = None,
                 use_class_weights: bool = False):

        super().__init__()

        self._identifier       = "binary-classifier"
        self._layer_dimensions = layer_dimensions
        self._layer_activation = layer_activation
        self._dropout          = dropout
        self._multilabel       = multilabel

        self._tracker = Tracker()

        self._embedder   = None
        self._model      = None
        self._label_dicts: dict[str, dict[int, Label]] = {}
        self._target_embeddings: dict[str, torch.FloatTensor] = {}

        if loss_functions:
            self._loss_functions = loss_functions
        else:
            if multilabel:
                self._loss_functions = [MultilabelBinaryLoss(positive_examples = 4,
                                                             negative_examples = 4,
                                                             use_class_weights = use_class_weights,
                                                             scale=1.0)]
            else:
                self._loss_functions = [BinaryLoss(use_class_weights=use_class_weights, scale=1.0)]

    def _initialize_head(self, input_dim: int):
        layers = []

        # Hidden layers
        prev_out_dim = input_dim
        for i, out_dim in enumerate(self._layer_dimensions):

            in_dim = prev_out_dim

            block = [('linear', torch.nn.Linear(in_dim, out_dim, bias=True))]

            if abs(self._dropout) > 1e-5:
                block.append(('dropout', torch.nn.Dropout(self._dropout)))

            block.append(('activation', self._layer_activation()))

            model_block = torch.nn.Sequential(OrderedDict(block))
            prev_out_dim = out_dim

            layers.append((f"block_{i}", model_block))

        # Output layer
        block = []
        if len(self._layer_dimensions):
            block.append(("output", torch.nn.Linear(prev_out_dim, 1, bias=True)))

        block.append(("output_activation", torch.nn.Sigmoid()))

        layers.append(("output", torch.nn.Sequential(OrderedDict(block))))

        self._model = torch.nn.Sequential(OrderedDict(layers))

        if self.is_cuda:
            self._change_device(cuda = True)

        if self.is_inference:
            self._change_mode(eval_mode = True)

    def use_identifier(self, identifier: str):
        self._identifier = identifier

    def forward(self,
                document_embeddings: torch.Tensor,
                target_embeddings: torch.Tensor = None) -> torch.Tensor:

        r"""
        Calculates prediction scores

        :param document_embeddings: documents x embedding_dim
        :param target_embeddings: documents x labels x embedding_dim (in inference: 1 x labels x embedding_dim )

        :return: scores
        """

        if target_embeddings is not None:
            x_in = target_embeddings * document_embeddings.unsqueeze(1)
        else:
            x_in = document_embeddings.unsqueeze(1)

        norm = x_in.norm(dim=2)
        x_in = (x_in / norm.masked_fill(norm < 1e-8, 1).unsqueeze(2))

        return self._model(x_in).squeeze(2)

    def use_loss_functions(self, loss_functions: list[LossStrategy]):
        self._loss_functions = loss_functions

    def get_losses(self, ctx: Context, documents: list[Document]) -> list[torch.Tensor]:
        losses = []
        for usecase in ctx.usecases:
            if usecase.name not in self.registered_usecase_names:
                continue

            for fn in self._loss_functions:
                losses.append(fn(model     = self,
                                 usecase   = usecase,
                                 documents = documents,
                                 embedder  = self._embedder))

        return losses

    # Pipeline methods
    def train(self,
              ctx:        Context = None,
              tokenizer:  Tokenizer  | ContextArg[Tokenizer]  = None,
              vectorizer: Vectorizer | ContextArg[Vectorizer] = None,
              embedder:   Embedder   | ContextArg[Embedder]   = None,
              **kwargs) -> None:
        r"""
        Trains the binary classifier

        The method accepts either an embedder or a tokenizer+vectorizer pair.

        :param ctx: optional Context
        :param tokenizer: instance of a trained tokenizer
        :param vectorizer: instance of a trained vectorizer
        :param embedder: instance of a trained embedder
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if self._model is not None:
            return

        if embedder and (tokenizer or vectorizer):
            raise Exception("Cannot depend on an embedder and tokenizer/vectorizer at the same time")

        elif not embedder and not (tokenizer and vectorizer):
            raise Exception("No tokenizer+vectorizer or embedder provided")

        if embedder:
            self._embedder = ContextArg.extract(embedder)
        else:
            self._embedder = DerivedEmbedder(tokenizer  = ContextArg.extract(tokenizer),
                                             vectorizer = ContextArg.extract(vectorizer))

        self._initialize_head(self._embedder.embedding_dim)

    def _prepare_target_embeddings(self,
                                   usecase_name: str,
                                   label_dict: dict[int, Label],
                                   batch_size: int = 128):

        self._label_dicts[usecase_name] = label_dict
        labels = list(label_dict.values())
        embs = [None] * len(labels)

        with torch.no_grad():
            for i in tqdm(range(0, len(labels), batch_size), "Preparing target embeddings"):
                text = labels[i:i+batch_size]
                embs[i:i+len(text)] = self._embedder.embed(text=text)

        # target_embedding shape: 1 x labels x embedder_dim
        self._target_embeddings[usecase_name] = torch.stack(embs, dim=0).unsqueeze(0)

    def _predict_multilabel(self,
                            ctx: Context,
                            label_dict: dict[int, Label] | ContextArg[dict[int, Label]],
                            document_embeddings: torch.Tensor,
                            max_predictions: int,
                            min_score: float) -> list[Prediction]:

        # Prepare target embeddings
        if (uname := ctx.active_usecase.name) not in self._target_embeddings:
            self._tracker.log("Preparing target embeddings")
            label_dict = ContextArg.extract(label_dict, ctx.active_usecase.name)
            self._prepare_target_embeddings(uname, label_dict)

        self._tracker.log("Transferring target embeddings to active device")
        target_embeddings = self.to_active_device(self._target_embeddings[uname])

        self._tracker.log("Calculating scores")
        scores = self.forward(document_embeddings, target_embeddings).detach().cpu().numpy()

        preds = []
        label_ids = list(self._label_dicts[uname].keys())
        label_values = list(self._label_dicts[uname].values())
        self._tracker.log(f"Starting to sort '{len(scores)}' document predictions")
        for i in range(len(scores)):
            self._tracker.log(f"Creating dataframe")
            df_pred = DataFrame({"label": label_values,
                                 "label_id": label_ids,
                                 "score": scores[i]}) \
                .sort_values(by="score", ascending=False) \
                .reset_index(drop=True)

            self._tracker.log(f"Creating raw payload")
            payload_raw = [RawPrediction(label=label_values[j],
                                         label_id=label_ids[j],
                                         score=s) for j, s in enumerate(scores[i])]

            self._tracker.log(f"Filtering dataframe")
            df_pred_final = df_pred \
                .query(f"score >= {min_score}") \
                .head(max_predictions) \
                .reset_index(drop=True)

            self._tracker.log(f"Creating prediction object")
            pred = Prediction(usecase_name=ctx.active_usecase.name,
                              model=self._identifier,
                              timestamp=int(datetime.now().timestamp()),
                              payload_raw=payload_raw,
                              payload=df_pred_final)

            self._tracker.log(f"Appending predictions to list")
            preds.append(pred)

            self._tracker.log(f"Done with document'{i}'")

        self._tracker.log("Finished predicting")

        return preds


    def _predict_binary(self,
                        ctx: Context,
                        document_embeddings: torch.Tensor,
                        binary_threshold: float) -> list[Prediction]:

        self._tracker.log("Calculating scores")
        scores = self.forward(document_embeddings).detach().cpu().numpy().squeeze(1)

        preds = []
        labels = ctx.active_usecase.get_usecase_labels()
        for score in scores:
            label_id = int(score >= binary_threshold)
            label    = labels[label_id]

            self._tracker.log(f"Creating raw payload")
            payload_raw = [RawPrediction(label = label,
                                         label_id=label_id,
                                         score=score)]

            df_pred = DataFrame({"label": labels,
                                 "score": [1-score, score],
                                 "selected": ["x" if score < binary_threshold else "",
                                              "x" if score >= binary_threshold else ""]})

            self._tracker.log(f"Creating prediction object")
            pred = Prediction(usecase_name=ctx.active_usecase.name,
                              model=self._identifier,
                              timestamp=int(datetime.now().timestamp()),
                              payload_raw=payload_raw,
                              payload=df_pred)

            preds.append(pred)

        return preds

    def predict(self,
                ctx:              Optional[Context],
                label_dict:       dict[int, Label] | ContextArg[dict[int, Label]] = None,
                embeddings:       torch.FloatTensor | ContextArg[torch.FloatTensor] = None,
                max_predictions:  int   = 10,
                min_score:        float = 0.0,
                binary_threshold: float = 0.5,
                **kwargs) -> list[Prediction]:

        r"""
        Calculates binary predictions

        Multilabel classification requires a label dictionary. The dictionary can be provided
        directly via predict method, or via ctx.active_usecase.get_usecase_label_dict()

        :param ctx: optional Context
        :param label_dict: label dictionary {label_id: label}
        :param embeddings: document embeddings
        :param max_predictions: max number of predictions (in multilabel case)
        :param min_score: prediction threshold  (in multilabel case)
        :param binary_threshold: binary threshold (in binary case)
        :param kwargs: optional, not specified list of keyword variables

        :return: list of predictions, one prediction per document
        """

        # Calculate scores
        with torch.no_grad():

            self._tracker.log("Preparing document embeddings")
            document_embeddings = ContextArg.extract(embeddings, ctx.active_usecase.name)
            document_embeddings = self.to_active_device(document_embeddings)

            if self._multilabel:
                label_dict = label_dict or ctx.active_usecase.get_usecase_label_dict()

                preds = self._predict_multilabel(ctx                 = ctx,
                                                 label_dict          = label_dict,
                                                 document_embeddings = document_embeddings,
                                                 max_predictions     = max_predictions,
                                                 min_score           = min_score)
            else:
                preds = self._predict_binary(ctx = ctx,
                                             document_embeddings = document_embeddings,
                                             binary_threshold    = binary_threshold)

        self._tracker.log("Finished predicting")

        return preds

    # Capability methods
    def get_cuda_components(self) -> list[Any]:
        return [self._model]

    def get_inferential_components(self) -> list[Any]:
        return self.get_cuda_components()

    def get_training_parameters(self) -> Any:
        if self._model is None:
            return []

        return [p for p in self._model.parameters() if p.requires_grad]

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    # Custom methods
    def use_label_dict(self, usecase_name: str, label_dict: dict[int, Label]):
        r"""
        Recalculates target embedings for multilabel classification

        :param usecase_name: active usecase name
        :param label_dict: label dictionary {label_id: label}

        :return: None
        """

        self._prepare_target_embeddings(usecase_name = usecase_name,
                                        label_dict   = label_dict)