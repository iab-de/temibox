from dataclasses import dataclass
from temibox.context import Context
from temibox.embedder import TFIDFEmbedder

from _support import MockMultiUseCase, MockDocumentMulti

@dataclass
class FakePipe:
    is_training: bool

#######################
# Setup
#######################

multi_documents = [
                   MockDocumentMulti(text="Dokument A und B", label_ids=list(range(10))),
                   MockDocumentMulti(text="Dokument B",       label_ids=list(range(5,12))),
                   MockDocumentMulti(text="Dokument A",       label_ids=list(range(20,25))),
                   MockDocumentMulti(text="Dokument A und C", label_ids=list(range(1, 7))),
                   MockDocumentMulti(text="Dokument A und D", label_ids=list(range(3,12))),
                   MockDocumentMulti(text="Dokument D",       label_ids=list(range(17,19))),
                   MockDocumentMulti(text="Dokument D und E", label_ids=list(range(21,31))),
                   MockDocumentMulti(text="Dokument D",       label_ids=list(range(5)))]*16

#######################
# Tests
#######################


def test_tfidf_train():

    labels = {x for d in multi_documents for x in d.label_ids}
    label_count = max(labels)
    uc_1 = MockMultiUseCase(use_labels={"a"*(i+1): (i,1) for i in range(label_count)})

    ctx = Context(FakePipe(is_training=False), [uc_1], uc_1)

    embedding_dim = 16

    emb = TFIDFEmbedder(max_tokens = 30_000,
                        embedding_dim = embedding_dim,
                        min_token_frequency = 1)

    emb.train(ctx, documents = multi_documents)

    assert emb._cuda_components[0].shape[0] == len(emb._tokens), "Invalid shape 0"
    assert emb._cuda_components[0].shape[1] == embedding_dim, "Invalid shape 1"

    embeddings = emb.transform(ctx = ctx, documents = multi_documents[:8])["embeddings"]

    assert embeddings.shape == (8, embedding_dim), "Embeddings shape is wrong"
    assert not embeddings.isnan().any().item(), "NANs are present"