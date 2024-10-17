import torch
import numpy as np
import pickle
from enum import Enum
from typing import Type
from dataclasses import dataclass

from ..domain import Document, UseCase, LabelDescription, Triplet, Label
from ..pipeline import StandardPipeline
from ..context import Context, ContextArg

from ..embedder import BertEmbedder
from ..model.classifier import BinaryClassifier, MultinomialClassifier
from ..trainer import StandardTrainer
from ..evaluation import Evaluator
from ..evaluation.metric import Accuracy, F1, ConfusionMatrix, Calibration, RocAuc, PrecisionAtK, RecallAtK

from .blueprint import Blueprint
from ..pipeline.pipeline import Pipeline
from ..prediction import Prediction
from ..tracker import Tracker

_CUDA = torch.cuda.is_available()


class TextClassification(Blueprint):
    r"""
    Blueprint for generic, BERT-based text classification.

    The classification task is chosen by providing a value of the TextClassification.TASK enum

    The blueprint document type can be referenced by TextClassification.Document
    The blueprint usecase type can be referenced by TextClassification.Classification
    """

    class TASK(Enum):
        r"""
        Text classification tasks recognized by the TextClassification blueprint
        """

        BINARY                 = "binary"                   # "Text" -> one of two labels
        MULTICLASS_MULTINOMIAL = "multiclass-multinomial"   # "Text" -> one of many labels (fixed set)
        MULTILABEL_BINARY      = "multilabel-binary"        # "Text" -> some of many labels (variable set)
        MULTILABEL_MULTINOMIAL = "multilabel-multinomial"   # "Text" -> some of many labels (fixed set)

    @dataclass
    class Document:
        r"""
        Generic document Type used in the blueprint Textclassification
        """
        text:      str
        label_ids: list[int] | None = None

    @dataclass
    class Classification(UseCase):
        r"""
        Textclassification usecase for blueprint TextClassification
        """

        def __init__(self,
                     labels: list[LabelDescription],
                     use_class_weights: bool = False):
            r"""
            Initializes the usecase

            :param labels: list of instances of LabelDescription describing the usecase's labels (ID, string, weight)
            :param use_class_weights: should (inverse) class weights be used in training
            """

            ld_dict = {ld.label_id: ld for ld in labels}
            self._use_class_weights = use_class_weights

            self._label_dict = {ld.label_id: ld.label for ld in labels}
            self._label_str = [ld.label for ld in ld_dict.values()]
            self._label_weights = {ld.label_id: ld.weight for ld in ld_dict.values()}
            self._label_str_weights = {self._label_dict[lid]: w for lid, w in self._label_weights.items()}

            minw = min(self._label_weights.values())
            inv_weights = {ld.label_id: minw/ld.weight for ld in ld_dict.values()}
            totw = sum(inv_weights.values())
            self._label_inv_weights = {k: v/totw for k, v in inv_weights.items()}

        @property
        def name(self) -> str:
            return "text-classification"

        def get_document_body(self, document: Document) -> str:
            return document.text

        def get_document_labels(self, document: Document) -> list[Label]:
            return [self._label_dict[lid] for lid in document.label_ids if lid in self._label_dict]

        def get_document_label_ids(self, document: Document) -> list[int]:
            return [lid for lid in document.label_ids if lid in self._label_dict]

        def get_usecase_labels(self) -> list[Label]:
            return self._label_str

        def get_usecase_label_dict(self) -> dict[int, Label]:
            return self._label_dict

        def get_usecase_label_weights(self) -> dict[Label, float]:
            return self._label_str_weights

        def create_document_test_labels(self, document: Document, positive: int, negative: int) -> list[tuple[Label, bool]]:
            ok_lids  = self.get_document_label_ids(document)
            bad_lids = list(self._label_dict.keys() - set(ok_lids))

            if self._use_class_weights:
                p_ok = [self._label_inv_weights[lid] for lid in ok_lids]
                ptot = sum(p_ok)
                p_ok = [p/ptot for p in p_ok]

                p_bad = [self._label_inv_weights[lid] for lid in bad_lids]
                ptot = sum(p_bad)
                p_bad = [p / ptot for p in p_bad]

                pos_lids = np.random.choice(ok_lids, positive, p=p_ok, replace=True).tolist()
                neg_lids = np.random.choice(bad_lids, negative, p=p_bad, replace=True).tolist()
            else:
                pos_lids = np.random.choice(ok_lids, positive, replace=True).tolist()
                neg_lids = np.random.choice(bad_lids, negative, replace=True).tolist()

            pos_labels = [(self._label_dict[lid], True) for lid in pos_lids]
            eng_labels = [(self._label_dict[lid], False) for lid in neg_lids]
            all_labels = pos_labels + eng_labels

            np.random.shuffle(all_labels)

            return all_labels

        def create_document_triplet(self, document: Document, examples: int = 1) -> list[Triplet]:
            labels  = sorted(self.create_document_test_labels(document, positive = examples, negative = examples), key = lambda x: x[1])
            neg_labels = [x[0] for x in labels[:examples]]
            pos_labels = [x[0] for x in labels[examples:]]
            anchor = document.text

            return [Triplet(p, anchor, n) for p,n in zip(pos_labels, neg_labels)]

    def __init__(self,
                 pretrained_bert_dir: str,
                 classification_task: TASK,
                 labels: list[LabelDescription],
                 documents: list[Document] = None,
                 use_class_weights: bool = False,
                 train_vali_split: float = 0.9,
                 layer_dimensions: list[int] = None,
                 allow_cuda: bool = True,
                 create_checkpoints: bool = True):
        r"""
        Initializes the TextClassification blueprint

        :param pretrained_bert_dir: directory containing a transformers-compatible pretrained bert model
        :param classification_task: value of the TextClassification.TASK enum
        :param labels: list of labels
        :param documents: optional list of documents. Training starts automatically if documents are provided (otherwise use blueprint.train)
        :param use_class_weights: should class weights be used in training
        :param train_vali_split: train/validation split used by trainer (checkpointing and early breaks are only available with train_vali_split < 0.99)
        :param layer_dimensions: layer dimensions used by the shallow classifier on top of the BERT-embedder (default: [32])
        :param allow_cuda: is the training process allowed to use the GPU?
        :param create_checkpoints: should checkpoints be created and potentially restored when training (only if validation data are available)
        """
        super().__init__()

        layer_dimensions = layer_dimensions if layer_dimensions is not None and len(layer_dimensions) else [32]

        self._pretrianed_path = pretrained_bert_dir
        self._labels          = labels.copy()
        self._usecase         = TextClassification.Classification(labels = labels,
                                                                  use_class_weights = use_class_weights)

        if classification_task == self.TASK.BINARY:
            model = BinaryClassifier(multilabel=False,
                                     layer_dimensions=layer_dimensions,
                                     use_class_weights = use_class_weights)
            metrics = [Accuracy(), F1(), RocAuc(), Calibration(), ConfusionMatrix()]

        elif classification_task == self.TASK.MULTILABEL_BINARY:
            model = BinaryClassifier(multilabel=True,
                                     layer_dimensions=layer_dimensions,
                                     use_class_weights = use_class_weights)
            metrics = [PrecisionAtK(k=5), PrecisionAtK(k=10), RecallAtK(k=5), RecallAtK(k=10), ConfusionMatrix()]

        elif classification_task == self.TASK.MULTICLASS_MULTINOMIAL:
            model = MultinomialClassifier(multilabel=False,
                                          layer_dimensions=layer_dimensions,
                                          use_class_weights = use_class_weights)
            metrics = [PrecisionAtK(k=5), PrecisionAtK(k=10), RecallAtK(k=5), RecallAtK(k=10), ConfusionMatrix()]

        elif classification_task == self.TASK.MULTILABEL_MULTINOMIAL:
            model = MultinomialClassifier(multilabel=True,
                                          layer_dimensions=layer_dimensions,
                                          use_class_weights = use_class_weights)
            metrics = [PrecisionAtK(k=5), PrecisionAtK(k=10), RecallAtK(k=5), RecallAtK(k=10), ConfusionMatrix()]

        else:
            raise Exception("Unsupported classification task")

        self._pipeline = StandardPipeline(allow_cuda = allow_cuda) \
                            .add_usecase(self._usecase) \
                            .add_step("embedder",  BertEmbedder(pretrained_model_dir=pretrained_bert_dir)) \
                            .add_step("model",     model) \
                            .add_step("trainer",   StandardTrainer(train_test_split=train_vali_split, create_checkpoints=create_checkpoints), dependencies=["embedder", "model"]) \
                            .add_step("evaluator", Evaluator(metrics=metrics))

        if documents is not None:
            self.train(documents = documents)

    def _get_ctx(self) -> Context:
        return Context(self._pipeline, [self._usecase], self._usecase)

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def usecase(self) -> UseCase:
        return self._usecase

    @property
    def document_type(self) -> Type[Document]:
        return TextClassification.Document

    def train(self,
              documents: list[Document] | ContextArg[list[Document]] = None,
              **kwargs):

        ctx = self._get_ctx()
        self._pipeline.set_inference_mode(on = False, cuda = _CUDA)
        self._pipeline.train(ctx = ctx, documents = documents, **kwargs)

    def predict(self,
                document: Document | list[Document] = None,
                min_score: float = 0.0,
                max_predictions: int = 10,
                batch_size: int = 32,
                **kwargs) -> list[Prediction]:
        r"""
        Runs the pipeline prediction workflow (pipeline.predict)

        :param document: optional document or list of documents
        :param min_score: minimum prediction score
        :param max_predictions: maximum number of predictions
        :param batch_size: batch size when predicting
        :param kwargs: other parameters possibly relevant to pipeline steps (see blueprint.pipeline.get_signature(Predictable))

        :return: list of predictions (one per document)
        """

        ctx = self._get_ctx()
        documents = document.copy() if isinstance(document, list) else [document]

        preds = []
        for i in range(0, len(documents), batch_size):
            preds += self._pipeline.predict(ctx = ctx,
                                          usecases = [self._usecase],
                                          documents = documents[i:i+batch_size],
                                          label_dict = self._usecase.get_usecase_label_dict(),
                                          min_score = min_score,
                                          max_predictions = max_predictions,
                                          **kwargs)
        return preds

    def evaluate(self,
                 documents: list[Document] | ContextArg[list[Document]] = None,
                 **kwargs) -> list[str]:

        ctx = self._get_ctx()
        metrics = ContextArg.extract(self._pipeline.evaluate(ctx = ctx,
                                          label_dict = self._usecase._label_dict,
                                          usecases = [self._usecase],
                                          documents = documents, **kwargs)["evaluator"]["metrics"])["model"]

        return list(metrics.values())

    def use_identifier(self, identifier: str) -> None:
        self._pipeline.use_identifier(identifier)

    def use_progress_tracker(self, tracker: Tracker) -> None:
        self._pipeline.use_progress_tracker(tracker)

    def get_progress_tracker(self) -> Tracker:
        return self._pipeline.get_progress_tracker()

    def export(self, full_path: str) -> None:
        r"""
        Cleans and exports the blueprint

        Cleaning the pipeline entails running the cleaning step on all of its cleanable steps,
        i.e. all the transient data and other variables are going to be removed and some workflows
        (e.g. training) might not be possible afterwards.

        :param full_path: full path (inkl. file extension) where the instance of the blueprint should be exported

        :return: None
        """
        self._pipeline.clean()
        with open(full_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Exported to {full_path}")

    @staticmethod
    def load(full_path: str) -> 'TextClassification':
        r"""
        Loads the exported instance

        :param full_path: full path (inkl. file extension) where the instance of the blueprint was exported

        :return: instance of TextClassification
        """
        with open(full_path, "rb") as f:
            obj = pickle.load(f)

        return obj
