import torch
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from typing import Optional
from datetime import datetime

from ...embedder.embedder import Embedder
from ...context import Context, ContextArg
from ...prediction import RawPrediction, Prediction
from ...traits import Trainable, Predictable
from ...domain import Document


class KnnClassifier(Trainable, Predictable):

    def __init__(self,
                 k_neighbours: int = 5,
                 bias_score: float = 0.5,
                 bias_terms: int = 1,
                 max_lookup_documents: int = -1):

        super().__init__()

        self._document_targets = {}
        self._document_embeddings = {}
        self._k_neighbours = k_neighbours if k_neighbours > 0 else 5
        self._identifier = "knn-classifier"

        self._bias_score = bias_score
        self._bias_terms = bias_terms
        self._max_lookup_documents = max_lookup_documents

    def train(self,
              ctx: Optional[Context] = None,
              embedder: Embedder | ContextArg[Embedder] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              **kwargs) -> None:
        r"""
        Trains the knn classifier

        :param ctx: optional Context
        :param embedder: instance of a trained embedder
        :param documents: list of lookup documents
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        if documents is None:
            raise Exception("No documents provided")

        documents = ContextArg.extract(documents)

        if self._max_lookup_documents > 0:
            documents = documents[:self._max_lookup_documents]

        for usecase in ctx.usecases:
            uname = usecase.name

            u_embedder = ContextArg.extract(embedder, uname)

            self._document_targets[uname] = [None]*len(documents)
            self._document_embeddings[uname] = torch.zeros((len(documents), u_embedder.embedding_dim), dtype=torch.float)

            with torch.no_grad():
                for i, doc in tqdm(enumerate(documents), f"Preparing KNN lookup table for usecase '{uname}'"):
                    try:
                        self._document_targets[uname][i]      = usecase.get_document_label_ids(doc)
                        self._document_embeddings[uname][i,:] = u_embedder.embed(text=usecase.get_document_body(doc)).detach()
                    except Exception as e:
                        print(f"Failed getting embedding of document {i}: {str(e)}")

                norm = self._document_embeddings[uname].norm(dim=1)
                self._document_embeddings[uname] /= norm.masked_fill(norm < 1e-8, 1).unsqueeze(1)

    def predict(self,
                ctx:             Optional[Context] = None,
                embeddings:      torch.FloatTensor | ContextArg[torch.FloatTensor] = None,
                max_predictions: int = 10,
                min_score:       float = 1e-5,
                **kwargs) -> list[Prediction]:
        r"""
        Classifies documents based on provided embeddings

        :param ctx: optional Context
        :param embeddings: document embeddings
        :param max_predictions: max number of predictions
        :param min_score: prediction threshold
        :param kwargs: optional, not specified list of keyword variables

        :return: list of predictions, one prediction per document
        """

        if embeddings is None:
            return []

        uname = ctx.active_usecase.name
        label_dict = ctx.active_usecase.get_usecase_label_dict()
        timestamp = int(datetime.now().timestamp())

        embeddings = ContextArg.extract(embeddings, uname)
        norm = embeddings.norm(dim=1)
        embeddings /= norm.masked_fill(norm < 1e-8, 1).unsqueeze(1)

        preds = []
        for i in range(embeddings.shape[0]):

            dist = torch.cdist(embeddings[i,].unsqueeze(0).cpu(), self._document_embeddings[uname]).detach()

            values = dist \
                        .squeeze(0) \
                        .numpy() \
                        .tolist()

            max_dist = max(1.25, max(values))

            idx = dist \
                    .topk(self._k_neighbours, largest=False, sorted=True) \
                    .indices \
                    .squeeze(0) \
                    .numpy() \
                    .tolist()

            df_dists = DataFrame()
            for j in idx:
                targets = self._document_targets[uname][j]
                targets = [tid for tid in targets if tid in label_dict]

                df_dists_i = DataFrame({ "label": [label_dict[tid] for tid in targets],
                                         "label_id": targets,
                                         "score": 1-values[j]/max_dist})

                df_dists = pd.concat([df_dists, df_dists_i], ignore_index=True)

            df_dists_agg = df_dists \
                .groupby(["label", "label_id"], as_index=False) \
                .agg(["max", "count"]) \
                .reset_index() \
                .droplevel(1, 1) \
                .set_axis(["label", "label_id", "score_max", "score_count"], axis=1) \
                .assign(score=lambda x: (x.score_max * x.score_count + self._bias_score * self._bias_terms) / (x.score_count + self._bias_terms)) \
                .drop(columns=["score_max", "score_count"]) \
                .sort_values(["score", "label_id"], ascending=[False, True])

            dists_raw   = {x["label_id"]: x["score"] for x in df_dists_agg[["label_id", "score"]].to_dict("records")}
            payload_raw = [RawPrediction(label_id=tid, label=label_dict[tid], score = dists_raw.get(tid, 0.0)) for tid in label_dict.keys()]

            payload = df_dists_agg \
                                .query(f"score >= {min_score}") \
                                .reset_index(drop=True) \
                                .head(max_predictions)

            pred = Prediction(usecase_name=ctx.active_usecase.name,
                              model=self._identifier,
                              timestamp=timestamp,
                              payload_raw=payload_raw,
                              payload=payload)

            preds.append(pred)

        return preds

    def use_identifier(self, identifier: str):
        self._identifier = identifier
