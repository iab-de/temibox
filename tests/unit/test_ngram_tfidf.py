from temibox.context import Context
from temibox.tokenizer import NgramTokenizer
from temibox.embedder import TFIDFEmbedder

from _support import MockDocument, MockUseCase


documents = [MockDocument(text="Das ist nur ein Test Text", label_id=0),
             MockDocument(text="Das ist auch ein Test Text", label_id=0),
             MockDocument(text="Test Text auch hier", label_id=0),
             MockDocument(text="Jedes Dokument enthält ein Test Text", label_id=1),
             MockDocument(text="Test Text soll sechsmal vorkommen", label_id=1),
             MockDocument(text="Mehr Test Text brauchen wir nicht", label_id=1)]

def test_ngram_tokenizer_train():
    tokenizer = NgramTokenizer(cased = False)
    uc = MockUseCase()

    ctx = Context(None, [uc], uc)

    tokenizer.train(ctx = ctx,
                    documents = documents,
                    stopwords = {"Das", "ist", "ein", "auch", "soll", "nicht", "wir"})

    assert tokenizer.get_token_set() == {'test', 'test text', 'text'}, "Token set is invalid"

    return tokenizer

def test_ngram_tokenizer_tokenize():

    uc = MockUseCase()
    ctx = Context(None, [uc], uc)
    tokenizer = test_ngram_tokenizer_train()

    out_1 = tokenizer.tokenize(text="Ich tokenisiere einen Test Text")
    out_2 = tokenizer.transform(text="Ich tokenisiere einen Test Text")

    assert out_1 == [['test', 'text', 'test text']], "Invalid 'tokenize' tokens"
    assert out_2 == {'tokens': [['test', 'text', 'test text']]}, "Invalid 'transform' tokens"

    for doc in documents:
        out = tokenizer.transform(ctx = ctx, document=doc)
        assert out == {'tokens': [['test', 'text', 'test text']]} or out == {'tokens': [['test', 'text']]}, "Invalid 'transform' tokens (doc loop)"

    out = tokenizer.transform(ctx = ctx, documents = documents)
    for tokens in out["tokens"]:
        assert tokens == ['test', 'text', 'test text'] or  tokens == ['test', 'text'], "Invalid 'transform' tokens (documents)"

def test_tfidf_embedder_train():
    uc = MockUseCase()
    ctx = Context(None, [uc], uc)
    embedder = TFIDFEmbedder(max_tokens=10, embedding_dim=16)

    embedder.train(ctx = ctx, documents = documents)