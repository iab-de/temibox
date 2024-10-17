from temibox.model.classifier import KMeansCluster
from temibox.context import Context, ContextArg


from _support import MockMultiUseCase, MockDocumentMulti, MockClasslessUseCase,MockEmbedder

#######################
# Setup
#######################

multi_documents = [
                   MockDocumentMulti(text="Dokument 1", label_ids=[1,2,3]),
                   MockDocumentMulti(text="Dokument 2", label_ids=[1,2]),
                   MockDocumentMulti(text="Dokument 3", label_ids=[1,3]),
                   MockDocumentMulti(text="Dokument 4", label_ids=[2,3]),
                   MockDocumentMulti(text="Dokument 5", label_ids=[1]),
                   MockDocumentMulti(text="Dokument 6", label_ids=[2]),
                   MockDocumentMulti(text="Dokument 7", label_ids=[3]),
                   MockDocumentMulti(text="Dokument 8", label_ids=[1,2,3])]*16


multi_nc_documents = [
                   MockDocumentMulti(text="Dokument 1", label_ids=[]),
                   MockDocumentMulti(text="Dokument 2", label_ids=[]),
                   MockDocumentMulti(text="Dokument 3", label_ids=[]),
                   MockDocumentMulti(text="Dokument 4", label_ids=[]),
                   MockDocumentMulti(text="Dokument 5", label_ids=[]),
                   MockDocumentMulti(text="Dokument 6", label_ids=[]),
                   MockDocumentMulti(text="Dokument 7", label_ids=[]),
                   MockDocumentMulti(text="Dokument 8", label_ids=[])]*16

def get_default_model(k_neighbours: int):

    uc_1 = MockMultiUseCase(use_labels={"a": (1,1), "b": (2,1), "c": (3,1)})

    ctx = Context(None, [uc_1], uc_1)

    m = KMeansCluster(min_clusters=2,
                      max_clusters=3,
                      cluster_step=1,
                      max_lookup_documents=-1)


    return uc_1, ctx, m

def test_kmeans_train():

    uc_1, ctx, m = get_default_model(k_neighbours=4)

    embedding_dim = 16

    m.train(ctx=ctx,
            embedder=MockEmbedder(embedding_dim=embedding_dim),
            documents=multi_documents)

    assert 2 <= m._k <= 3, "Optimal k is wrong"
    assert len(m._cluster_labels[uc_1.name]) == m._k, "Number of clusters is wrong"
    assert m._k == m._model[uc_1.name].n_clusters, "Number of clusters is wrong in the underlying model"


def test_kmeans_predict():

    uc_1, ctx, m = get_default_model(k_neighbours=4)

    embedding_dim = 32
    embedder = MockEmbedder(embedding_dim=embedding_dim)

    m.train(ctx=ctx,
            embedder=embedder,
            documents=multi_documents)

    pred_docs = multi_documents * 10

    embeddings = embedder.transform(ctx=ctx,
                                    documents=pred_docs)["embeddings"]

    assert embeddings.shape == (len(pred_docs), embedding_dim), "Invallid embedding shape"

    predictions = m.predict(ctx, embeddings=embeddings)
    assert len(predictions) == len(pred_docs), "Wrong number of predictions"
    assert list(predictions[0].payload.columns) == ["label_id", "label", "cluster", "score"], "Wrong payload"

def test_kmeans_no_class():

    uc_1 = MockClasslessUseCase()

    ctx = Context(None, [uc_1], uc_1)

    m = KMeansCluster(min_clusters=2,
                      max_clusters=3,
                      cluster_step=1,
                      max_lookup_documents=-1)

    embedding_dim = 32
    embedder = MockEmbedder(embedding_dim=embedding_dim)

    m.train(ctx=ctx,
            embedder=embedder,
            documents=multi_nc_documents)

    embeddings = embedder.transform(ctx=ctx,
                                    documents=multi_nc_documents)["embeddings"]

    assert embeddings.shape == (len(multi_nc_documents), embedding_dim), "Invallid embedding shape"

    predictions = m.predict(ctx, embeddings=embeddings)

    assert len(predictions) == len(multi_nc_documents), "Wrong number of predictions"
    assert list(predictions[0].payload.columns) == ["cluster"], "Wrong payload"