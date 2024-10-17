import torch
from pandas import DataFrame
from collections import OrderedDict
from typing import Optional, Any, Type
from datetime import datetime

from .classifier import Classifier

from ...tokenizer.tokenizer import Tokenizer
from ...vectorizer.vectorizer import Vectorizer
from ...embedder.embedder import Embedder
from ...embedder.derived_embedder import DerivedEmbedder
from ...traits import Trackable
from ...capabilities import CudaCapable, InferenceCapable, ParameterCapable
from ...context import Context, ContextArg
from ...domain import Document, Label
from ...prediction import Prediction, RawPrediction
from ...tracker import Tracker
from ...losses import LossStrategy, MultinomialLoss


class MultinomialClassifier(Classifier, CudaCapable, InferenceCapable, ParameterCapable, Trackable):
    r"""
    Multinomial classifier to be used as a shallow network on top of a transformer-based embedder.

    Can be used for multinomial and multilabel use cases
    """

    def __init__(self,
                 multilabel: bool,
                 layer_dimensions: list[int],
                 layer_activation: Type[torch.nn.Module] = torch.nn.ReLU,
                 dropout: float = 0.0,
                 loss_functions: list[LossStrategy] = None,
                 use_class_weights: bool = False):

        super().__init__()

        self._multilabel = multilabel
        self._identifier = "multinomial-classifier"
        self._tracker = Tracker()

        self._layer_dimensions = layer_dimensions
        self._layer_activation = layer_activation
        self._dropout = dropout

        self._embedder = None
        self._label_dicts: dict[str, dict[int, Label]] = {}

        self._common_model = None
        self._heads = {}

        if loss_functions:
            self._loss_functions = loss_functions
        else:
            self._loss_functions = [MultinomialLoss(use_class_weights=use_class_weights, scale=1.0)]

    def _initialize_head(self,
                         input_dim: int,
                         output_dims: dict[str, int]):
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

        self._common_model = torch.nn.Sequential(OrderedDict(layers))

        # Output heads
        self._heads = {}
        for name, out_dim in output_dims.items():
            block = []
            if len(self._layer_dimensions):
                block.append(("output", torch.nn.Linear(prev_out_dim, out_dim, bias=True)))

            if self._multilabel:
                block.append(("output_activation", torch.nn.Sigmoid()))
            else:
                block.append(("output_activation", torch.nn.Softmax(dim=1)))

            self._heads[name] = torch.nn.Sequential(OrderedDict(block))

        if self.is_cuda:
            self._change_device(cuda = True)

        if self.is_inference:
            self._change_mode(eval_mode = True)

    def forward(self,
                usecase_name: str,
                document_embeddings: torch.Tensor) -> torch.Tensor:

        r"""
        Calculates prediction scores

        :param usecase_name: name of the active usecase
        :param document_embeddings: documents x embedding_dim

        :return: scores
        """

        x_in = document_embeddings
        x_in = (x_in / x_in.norm(dim=1).unsqueeze(1))

        x_common = self._common_model(x_in)

        return self._heads[usecase_name](x_common)

    def use_loss_functions(self, loss_functions: list[LossStrategy]):
        self._loss_functions = loss_functions

    def get_losses(self, ctx: Optional[Context], documents: list[Document]) -> list[torch.Tensor]:
        losses = []
        for usecase in ctx.usecases:
            if usecase.name not in self.registered_usecase_names:
                continue

            for fn in self._loss_functions:
                losses.append(fn(model=self,
                                 usecase=usecase,
                                 documents=documents,
                                 embedder=self._embedder))

        return losses

    def use_identifier(self, identifier: str):
        self._identifier = identifier

    # Capability methods
    def get_cuda_components(self) -> list[Any]:
        return [self._common_model] + list(self._heads.values())

    def get_inferential_components(self) -> list[Any]:
        return self.get_cuda_components()

    def get_training_parameters(self) -> Any:
        if self._common_model is None:
            return []

        return [p for c in self.get_cuda_components() for p in c.parameters() if p.requires_grad]

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._tracker = tracker

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    # Pipeline methods
    def train(self,
              ctx: Context = None,
              tokenizer: Tokenizer | ContextArg[Tokenizer] = None,
              vectorizer: Vectorizer | ContextArg[Vectorizer] = None,
              embedder: Embedder | ContextArg[Embedder] = None,
              **kwargs) -> None:

        r"""
        Trains the multinomial classifier

        The method accepts either an embedder or a tokenizer+vectorizer pair.

        :param ctx: optional Context
        :param tokenizer: instance of a trained tokenizer
        :param vectorizer: instance of a trained vectorizer
        :param embedder: instance of a trained embedder
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if self._common_model is not None:
            return

        if embedder and (tokenizer or vectorizer):
            raise Exception("Cannot depend on an embedder and tokenizer/vectorizer at the same time")

        elif not embedder and not (tokenizer and vectorizer):
            raise Exception("No tokenizer+vectorizer or embedder provided")

        if embedder:
            self._embedder = ContextArg.extract(embedder)
        else:
            self._embedder = DerivedEmbedder(tokenizer=ContextArg.extract(tokenizer),
                                             vectorizer=ContextArg.extract(vectorizer))

        output_dims = {u.name: len(u.get_usecase_labels()) for u in self.registered_usecases}
        self._initialize_head(self._embedder.embedding_dim, output_dims)

    def predict(self,
                ctx: Optional[Context],
                label_dict: dict[int, Label] | ContextArg[dict[int, Label]] = None,
                embeddings: torch.FloatTensor | ContextArg[torch.FloatTensor] = None,
                max_predictions: int = 10,
                min_score: float = 0.0,
                binary_threshold: float = 0.5,
                **kwargs) -> list[Prediction]:
        r"""
        Calculates multinomial predictions

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

            self._tracker.log("Calculating scores")
            scores = self.forward(usecase_name =  ctx.active_usecase.name,
                                  document_embeddings = document_embeddings).detach().cpu().numpy()

            preds = []
            label_dict = label_dict or ctx.active_usecase.get_usecase_label_dict()
            label_ids = list(label_dict.keys())
            label_values = list(label_dict.values())
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

        return preds
