import torch
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
from typing import Optional

from ...traits import Trainable, Predictable
from ...embedder.embedder import Embedder
from ...prediction import RawPrediction, Prediction
from ...context import Context, ContextArg
from ... domain import Document


class KMeansCluster(Trainable, Predictable):

    def __init__(self,
                 min_clusters: int,
                 max_clusters: int,
                 cluster_step: int,
                 max_lookup_documents: int = -1):
        r"""
        K-means Clustering model

        :param min_clusters: minimum number of clusters used in training (>=2)
        :param max_clusters: maximum number of clusters used in training
        :param cluster_step: cluster step size when looking for optimal k (usually = 1)
        :param max_lookup_documents: maximum number of documents to use for training
        """

        super().__init__()

        self._model = {}
        self._cluster_labels = {}
        self._k = -1
        self._identifier = "kmeans-cluster"

        self._max_lookup_documents = max_lookup_documents
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._cluster_step = cluster_step

    def train(self,
              ctx: Optional[Context] = None,
              embedder: Embedder | ContextArg[Embedder] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              train_split: float = 0.9,
              **kwargs) -> None:

        r"""
        Trains the kmeans clusterer / classifier

        :param ctx: optional Context
        :param embedder: instance of a trained embedder
        :param documents: list of lookup documents
        :param train_split: training / validation split used to determine the optimal number of clusters
        :param kwargs: optional, not specified list of keyword variables

        :return: None
        """

        # Max Lookup documents
        for usecase in ctx.usecases:
            uname = usecase.name
            label_dict = usecase.get_usecase_label_dict()

            embedder = ContextArg.extract(embedder, uname)

            documents = ContextArg.extract(documents, uname)
            max_docs = self._max_lookup_documents if self._max_lookup_documents > 0 else len(documents)
            documents = documents[:max_docs]

            # Embeddings
            embeddings = torch.zeros((len(documents), embedder.embedding_dim), dtype=torch.float)
            with torch.no_grad():
                for i, doc in tqdm(enumerate(documents), f"Preparing kMeans embeddings for usecase '{uname}'"):
                    embeddings[i,:] = embedder.embed(text = usecase.get_document_body(doc)).detach()

            norm = embeddings.norm(dim=1)
            embeddings /= norm.masked_fill(norm < 1e-8, 1).unsqueeze(1)

            # Optimal k
            inertias = []
            silhouette_scores = []
            x = embeddings.nan_to_num(0).numpy()
            min_k = self._min_clusters
            max_k = min(x.shape[0]-2, self._max_clusters)
            for k in tqdm(range(min_k, max_k+1, self._cluster_step), "Learning optimal 'k' for k-means"):
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(x)
                labels = kmeans.predict(x)

                inertias.append((k, kmeans.inertia_))
                silhouette_scores.append((k, silhouette_score(x, labels)))

            idx = np.argmax([s[1] for s in silhouette_scores])
            self._k = silhouette_scores[idx][0]

            # Final Model
            self._model[uname] = KMeans(n_clusters=self._k, random_state=0)
            self._model[uname].fit(x)

            # Cluster labels
            yhats = self._model[uname].predict(x)
            self._cluster_labels[uname] = {}
            for i, doc in enumerate(documents):
                cluster = yhats[i]
                if cluster not in self._cluster_labels[uname]:
                    self._cluster_labels[uname][cluster] = Counter()

                self._cluster_labels[uname][cluster].update([label_id for label_id in usecase.get_document_label_ids(doc) if label_id in label_dict])

    def predict(self,
                ctx: Optional[Context] = None,
                embeddings: torch.FloatTensor | ContextArg[torch.FloatTensor] = None,
                max_predictions: int = 10,
                min_score: float = 1e-5,
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

        clusters = self._model[uname].predict(embeddings.detach().cpu().numpy())

        assert clusters.shape[0] == embeddings.shape[0], "Prediction error"

        preds = []
        for i in range(embeddings.shape[0]):
            cluster = clusters[i]
            cluster_labels = self._cluster_labels[uname].get(cluster, None)

            if not cluster_labels:
                label_ids = [None]
                scores    = [None]
                preds_i   = {}
            else:
                label_ids = [x[0] for x in cluster_labels.most_common(max_predictions)]
                scores = [x[1] for x in cluster_labels.most_common(max_predictions)]
                scores = [s/max(scores) for s in scores]

                preds_i = {lid: score for lid, score in zip(label_ids, scores)}

            payload_raw = [RawPrediction(label_id = lid,
                                         label    = label,
                                         score    = preds_i.get(lid, 0.0)) for lid, label in label_dict.items()]

            payload = DataFrame({"label_id": label_ids,
                                 "label": [label_dict.get(lid) for lid in label_ids],
                                 "cluster": [cluster]*len(label_ids),
                                 "score": scores})

            if not cluster_labels:
                payload = payload[["cluster"]]
            else:
                payload = (payload
                            .query(f"score >= {min_score}")
                            .sort_values(["score", "label_id"], ascending = [False, True])
                            .reset_index(drop = True)
                            .head(max_predictions))

            pred = Prediction(usecase_name=ctx.active_usecase.name,
                              model=self._identifier,
                              timestamp=timestamp,
                              payload_raw=payload_raw,
                              payload=payload)

            preds.append(pred)

        return preds

    def use_identifier(self, identifier: str):
        self._identifier = identifier