from tqdm import tqdm
from typing import Any, Optional

from ..domain import UseCase, Document
from ..traits import Evaluating, Supervising, Predictable
from ..context import Context, ContextArg

from .metric.metric import Metric


class Evaluator(Evaluating, Supervising):
    r"""
    Pipeline performance evaluator

    Calculates a list of metrics for a given set of documents and a pipeline
    """

    def __init__(self, metrics: list[Metric]):
        super().__init__()

        self._metrics = metrics
        self._models: dict[str, Predictable] = {}

    def evaluate(self,
                  ctx: Optional[Context] = None,
                  documents: list[Document] | ContextArg[list[Document]] = None,
                  batch_size: int = 64,
                  **kwargs) -> dict[str, Any]:
        r"""
        Evaluates pipeline performance for a given set of documents and metrics

        :param ctx: optional Context
        :param documents: list of (test-)documents
        :param batch_size: batch size when predicting
        :param kwargs: optional, not specified list of keyword variables

        :return: dictionary of dictionaries of metrics per predictable step
        """

        # Only evaluate after training
        if ctx.pipeline.is_training:
            return {}

        documents = ContextArg.extract(documents, ctx.active_usecase.name)

        if documents is None or not len(documents):
            raise Exception("No documents provided")

        # Predict in batches
        predictions = []
        for i in tqdm(range(0, len(documents), batch_size), "Evaluating"):
            docs  = documents[i:i+batch_size]
            preds = ctx.pipeline.predict(ctx=ctx, documents=docs, **kwargs)
            reps  = len(preds)//len(docs)
            assert len(preds) == len(docs)*reps, "Number of documents and predictions don't match"

            predictions += zip(docs * reps, preds)

        raw_predictions_by_model = {x[1].model: ([], []) for x in predictions if x[1].model}
        for doc, prediction in predictions:
            if ctx.active_usecase.name != prediction.usecase_name:
                continue

            raw_predictions_by_model[prediction.model][0].append(doc)
            raw_predictions_by_model[prediction.model][1].append(prediction.payload_raw)


        # Filter out supervised models
        raw_predictions_by_model = {k: v for k,v in raw_predictions_by_model.items() if k in self.supervising()}

        # Evaluate
        results = {k: {} for k in raw_predictions_by_model.keys()}
        for model_name, (docs, preds) in raw_predictions_by_model.items():
            if len(docs) != len(preds):
                raise Exception("Number of documents does not match number of predictions")

            for i, metric in enumerate(self._metrics):
                try:
                    results[model_name][metric.name] = metric(raw_predictions = preds,
                                                              usecase         = ctx.active_usecase,
                                                              documents       = docs,
                                                              **kwargs)
                except Exception as e:
                    print(e)

        return {"metrics": results}

    def supervise(self, name: str, step: Predictable, **kwargs) -> None:
        r"""
        Adds a predictable steps to its set of monitored steps

        :param name: name of the step
        :param step: predictable step
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if isinstance(step, Predictable):
            self._models[name] = step

    def supervising(self) -> list[str]:
        r"""
        Returns the list of supervised model names

        :return: list of names
        """
        return list(self._models.keys())

    def use_metrics(self, metrics: list[Metric]):
        r"""
        Uses the list of provided metrics instead of the metrics provided on
        initialization

        :param metrics: list of metrics

        :return: None
        """

        self._metrics = metrics