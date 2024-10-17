from temibox.model.classifier import KnnClassifier
from temibox.context import Context, ContextArg


from _support import MockMultiUseCase, MockDocumentMulti, MockEmbedder

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

def get_default_model(k_neighbours: int):

    uc_1 = MockMultiUseCase(use_labels={"a": (1,1), "b": (2,1), "c": (3,1)})

    ctx = Context(None, [uc_1], uc_1)

    m = KnnClassifier(k_neighbours=k_neighbours,
                      bias_score=0.5,
                      bias_terms=1,
                      max_lookup_documents=-1)


    return uc_1, ctx, m

#######################
# Tests
#######################

def test_knn_train():

    uc_1, ctx, m = get_default_model(k_neighbours = 4)

    embedding_dim = 16

    m.train(ctx = ctx,
            embedder  = MockEmbedder(embedding_dim = embedding_dim),
            documents = multi_documents)

    assert uc_1.name in m._document_targets, "Usecase missing in document targets"
    assert {c for x in m._document_targets[uc_1.name] for c in x} == set(uc_1.get_usecase_label_dict()), "Document targets are wrong"

    assert uc_1.name in m._document_embeddings, "Usecase missing in document embeddings"
    assert m._document_embeddings[uc_1.name].shape == (len(multi_documents), embedding_dim)


def test_knn_predict():

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

    for p in predictions:
        assert len(p.payload_raw) == len(uc_1.get_usecase_labels()), "Number of raw predictions does not match the number of labels"
        assert {r.label for r in p.payload_raw} == set(uc_1.get_usecase_labels()), "Not all labels present in raw prediction"