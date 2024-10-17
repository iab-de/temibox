import os
import pytest
from dataclasses import dataclass

from temibox.tokenizer import BertTokenizer
from temibox.vectorizer import BertVectorizer
from temibox.embedder import DerivedEmbedder, BertEmbedder
from temibox.tracker import Tracker
from temibox.context import Context, ContextArg

from _support import MockDocument, MockUseCase

#######################
# Setup
#######################
@dataclass
class FakePipeline:
    is_training: bool

PRETRAINED_DIR = os.getenv("PRETRAINED_DIR")


documents = [MockDocument(text="Dokument 1", label_id=0),
             MockDocument(text="Dokument 2", label_id=0),
             MockDocument(text="Dokument 3", label_id=0),
             MockDocument(text="Dokument 4", label_id=1),
             MockDocument(text="Dokument 5", label_id=1),
             MockDocument(text="IAB sollte ein unbekannter Token sein", label_id=1)]

#######################
# Tests
#######################

def test_pretrained_dir():
    assert os.path.isdir(PRETRAINED_DIR), "Pretrained directory does not exist"

def test_bert_tokenizer():
    tokenizer = BertTokenizer(pretrained_model_dir = PRETRAINED_DIR)

    out = tokenizer.tokenize(text="Das ist ein Textbeispiel")

    assert len(out.keys() & {"input_ids", "attention_mask"}) == 2, "Tokenizer does not return expected data"

def test_bert_tokenizer_train_transform():
    usecase_1 = MockUseCase(name = "usecase-1")
    usecase_2 = MockUseCase(name="usecase-2")
    ctx = Context(None, [usecase_1, usecase_2], usecase_1)

    tokenizer = BertTokenizer(pretrained_model_dir = PRETRAINED_DIR)
    tokenizer.register_usecases([usecase_1])

    with pytest.raises(Exception):
        tokenizer.train()

    with pytest.raises(Exception):
        tokenizer.train(ctx = ctx, documents = None, stopwords = {"3", "4", "5"})

    tokenizer.train(ctx = ctx, documents = documents, stopwords = {"3", "4", "5"})
    tokenizer.train() # Early return

    with pytest.raises(Exception):
        tokenizer.transform(ctx=None, document=ContextArg(usecase_1.name, documents[0]))

    with pytest.raises(Exception):
        tokenizer.transform(ctx=ctx)

    out_1 = tokenizer.transform(ctx = ctx, document = ContextArg(usecase_1.name, documents[0]))
    out_2 = tokenizer.transform(ctx = ctx, documents = ContextArg(usecase_1.name, documents))

    assert out_1.keys() == out_2.keys() == {"tokenizer", "vocab_size", "texts", "tokens"}, "Transform output is wrong"
    assert len(out_2["tokens"].keys() & {"input_ids", "attention_mask"}) == 2,  "Tokenizer does not return expected data"
    assert out_2["tokens"]["input_ids"].shape[0] == len(documents), "Token shape invalid"

    ctx = Context(None, [usecase_1, usecase_2], usecase_2)
    out = tokenizer.transform(ctx = ctx, document = ContextArg(usecase_1.name, documents[0]))
    assert len(out) == 0, "Should be empty"

def test_bert_tokenizer_transform_is_training():
    usecase = MockUseCase(name = "usecase")
    ctx = Context(FakePipeline(is_training=True), [usecase], usecase)

    tokenizer = BertTokenizer(pretrained_model_dir = PRETRAINED_DIR)
    tokenizer.register_usecases([usecase])

    out = tokenizer.transform(ctx=ctx, document = documents[0])

    assert len(out) == 2, "Too many outputs"
    assert out.keys() == {"tokenizer", "vocab_size"}

def test_bert_tokenizer_add_tokens():
    usecase = MockUseCase()
    ctx = Context(None, [usecase], usecase)

    tokenizer = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR, min_new_token_freq=1)
    tokenizer.register_usecases([usecase])
    tokens_before = tokenizer.get_token_set().copy()
    tokenizer.train(ctx=ctx, documents=documents)
    tokens_after = tokenizer.get_token_set().copy()
    token_ids = tokenizer.get_token_ids()

    assert tokenizer._added_tokens > 0, "There should be some unknown tokens"
    assert len(tokens_after) - len(tokens_before) == tokenizer._added_tokens, "Not all tokens added"
    assert len(tokens_after) == len(token_ids), "Token set and token id set not matching"

    assert "IAB" not in tokens_before, "Token IAB is unknown"
    assert "IAB" in tokens_after, "Token IAB is known"

def test_bert_tokenizer_add_no_tokens():
    usecase = MockUseCase()
    ctx = Context(None, [usecase], usecase)

    tokenizer = BertTokenizer(pretrained_model_dir = PRETRAINED_DIR,
                              allow_max_new_tokens = 0,
                              min_new_token_freq = 1)

    tokenizer.register_usecases([usecase])
    tokenizer.train(ctx=ctx, documents=documents)
    tokens_after = tokenizer.get_token_set().copy()

    assert tokenizer._added_tokens == 0, "Should not add new tokens"

    assert "IAB" not in tokens_after, "Token IAB is known"

def test_bert_tokenizer_tokenize():

    tokenizer = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer.configure_cache(on = True)

    out_1 = tokenizer.tokenize(text=["Unit Test"])
    out_2 = tokenizer.tokenize(text="Unit Test2")
    out_3 = tokenizer.tokenize(text=("Unit", "Test2"))

    assert len(out_1.keys() & {"input_ids", "attention_mask"}) == 2, "Tokenizer does not return expected data"
    assert len(out_2.keys() & {"input_ids", "attention_mask"}) == 2, "Tokenizer does not return expected data"
    assert len(out_3.keys() & {"input_ids", "attention_mask"}) == 2, "Tokenizer does not return expected data"

def test_bert_tokenizer_cache():

    tokenizer = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)
    assert not tokenizer.is_caching, "Should not be caching"

    tokenizer.configure_cache(on = True, max_entries = 1024)
    assert tokenizer.is_caching, "Should  be caching"
    assert tokenizer.cache.size == 0, "Cache should be empty"

    tokenizer.tokenize(text=["Unit Test"])
    tokenizer.tokenize(text=["Unit Test"])
    tokenizer.tokenize(text=["Unit Test"])
    assert tokenizer.cache.size == 1, "Cache should not be empty"

    tokenizer.clean()
    assert tokenizer.cache.size == 0, "Cache should be empty"


def test_bert_vectorizer_train():
    usecase_1 = MockUseCase(name="usecase-1")
    usecase_2 = MockUseCase(name="usecase-2")
    ctx = Context(None, [usecase_1, usecase_2], usecase_1)

    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer  = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)

    with pytest.raises(Exception):
        vectorizer.train(ctx=None, vocab_size=len(tokenizer.get_token_ids()))

    with pytest.raises(Exception):
        vectorizer.train(ctx=ctx, vocab_size=None)

    assert not vectorizer._is_trained, "Vectorizer should not be trained"
    vectorizer.train(ctx = ctx, vocab_size=len(tokenizer.get_token_ids()))
    assert vectorizer._is_trained, "Vectorizer should be trained"
    vectorizer.train(ctx = ctx, vocab_size=len(tokenizer.get_token_ids())) # return early

def test_bert_vectorizer_increase_vocab():
    usecase = MockUseCase(name="usecase-1")
    ctx = Context(None, [usecase], usecase)

    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer  = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)

    vectorizer.train(ctx = ctx, vocab_size=len(tokenizer.get_token_ids()) + 1)
    assert vectorizer._is_trained, "Vectorizer should be trained"

def test_bert_vectorizer_vocab_size():
    usecase = MockUseCase(name="usecase-1")
    ctx = Context(None, [usecase], usecase)

    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer  = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)

    vectorizer.register_usecases([usecase])
    vectorizer.train(ctx = ctx, vocab_size=ContextArg(usecase.name, len(tokenizer.get_token_ids())))
    assert vectorizer._is_trained, "Vectorizer should be trained"

def test_bert_vectorizer_transform():
    usecase_1 = MockUseCase(name="usecase-1")
    usecase_2 = MockUseCase(name="usecase-2")
    ctx_1 = Context(FakePipeline(is_training=False), [usecase_1, usecase_2], usecase_1)
    ctx_2 = Context(FakePipeline(is_training=False), [usecase_1, usecase_2], usecase_2)

    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer  = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)

    tokenizer.register_usecases([usecase_1])
    vectorizer.register_usecases([usecase_1])

    vectorizer.configure_cache(on=True)

    tokenizer.train(ctx = ctx_1, documents=documents)
    vectorizer.train(ctx = ctx_1, vocab_size=len(tokenizer.get_token_ids()) + 1)
    assert vectorizer._is_trained, "Vectorizer should be trained"

    text = ["Unit Test", "Unit Test 2"]
    tokens = tokenizer.tokenize(text = text)

    with pytest.raises(Exception):
        vectorizer.transform(ctx = None, texts = text, tokens = tokens)

    assert vectorizer.transform(ctx = ctx_2, texts = text, tokens = tokens) == {}, "Should be empty"

    with pytest.raises(Exception):
        vectorizer.transform(ctx=ctx_1, texts=text, tokens=None)

    with pytest.raises(Exception):
        vectorizer.transform(ctx=ctx_1, texts=None, tokens=None)

    out_1 = vectorizer.transform(ctx = ctx_1, texts = text, tokens = tokens)
    out_2 = vectorizer.transform(ctx = ctx_1, texts = None, tokens = tokens)

    for out in [out_1, out_2]:
        assert len(out) == 2, "Vectorizer output should have two elements"
        assert len(out.keys() & {"vectorizer", "embeddings"}) == 2, "Vectorizer output should contain self and embeddings"

        assert out["embeddings"].shape[0] == len(text), "Embeddings - incorrect shape[0]"
        assert out["embeddings"].shape[1] == vectorizer.embedding_dim, "Embeddings - incorrect shape[0]"

    ctx_1.pipeline.is_training = True
    out_3 = vectorizer.transform(ctx = ctx_1, texts = text, tokens = tokens)

    assert len(out_3) == 1, "Vectorizer output should have one element in training"
    assert out_3.keys() == {"vectorizer"}, "Vectorizer output should contain self and embeddings"

def test_bert_vectorizer_vectorize():
    usecase = MockUseCase(name="usecase-1")
    ctx = Context(FakePipeline(is_training=False), [usecase], usecase)

    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tokenizer  = BertTokenizer(pretrained_model_dir=PRETRAINED_DIR)

    tokenizer.register_usecases([usecase])
    vectorizer.register_usecases([usecase])

    vectorizer.configure_cache(on=True)
    assert vectorizer.is_caching, ""

    tokenizer.train(ctx = ctx, documents=documents)
    vectorizer.train(ctx = ctx, vocab_size=len(tokenizer.get_token_ids()) + 1)
    assert vectorizer._is_trained, "Vectorizer should be trained"

    text = ["Unit Test", "Unit Test 2"]
    tokens = tokenizer.tokenize(text = text)

    assert vectorizer.cache.size == 0, "Cache should be empty"
    out_1 = vectorizer.transform(ctx = ctx, texts = text, tokens = tokens)
    assert vectorizer.cache.size == len(text), "Cache should not be empty"
    for _ in range(5):
        out_2 = vectorizer.transform(ctx=ctx, texts=text, tokens = tokens)
        assert vectorizer.cache.size == len(text), "Cache should not be empty"

    assert out_1["embeddings"].allclose(out_2["embeddings"]), "All outputs should be equal"

    vectorizer.clean()
    assert vectorizer.cache.size == 0, "Cache should be empty"

def test_bert_components():
    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    cuda = vectorizer.get_cuda_components()
    infr = vectorizer.get_inferential_components()

    assert cuda == infr, "Components should match"

def test_bert_params():
    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)

    assert not vectorizer.is_inference, "Should not be in inference mode"
    assert len( vectorizer.get_training_parameters()) > 0, "Parameters should not be empty"

    vectorizer.set_inference_mode(on = True)
    assert vectorizer.is_inference, "Shouild be in inference mode"
    assert len(vectorizer.get_training_parameters()) == 0, "Parameters should be empty"

def test_bert_tracker():
    vectorizer = BertVectorizer(pretrained_model_dir=PRETRAINED_DIR)
    tracker = Tracker()

    vectorizer.use_progress_tracker(tracker)
    assert vectorizer.get_progress_tracker() == tracker, "Tracker instances should match"

def test_bert_embedder():
    usecase = MockUseCase(name="usecase-1")
    ctx = Context(FakePipeline(is_training=False), [usecase], usecase)

    embedder = BertEmbedder(pretrained_model_dir=PRETRAINED_DIR)
    embedder.register_usecases([usecase])

    embedder.train(ctx=ctx, documents=documents)
    out_1 = embedder.transform(ctx = ctx, documents=documents)

    keys = {"embedder", "tokens", "embeddings", "texts"}
    assert out_1.keys() & keys == keys, "All keys included"

    text = ["Unit Test", "Unit Test 2"]
    out_2 = embedder.embed(text=text)

    assert out_2.shape[0] == len(text), "Embeddings shape[0] wrong"
    assert out_2.shape[1] == embedder.embedding_dim, "Embeddings shape[1] wrong"

def test_embedder_cache():
    embedder = BertEmbedder(pretrained_model_dir=PRETRAINED_DIR)
    embedder.configure_cache(on=True)
    assert embedder.is_caching, "Should be caching"
    assert embedder.cache.size == 0, "Cache should be empty"

    text = ["Unit Test", "Unit Test 2"]
    embedder.embed(text=text)
    embedder.embed(text=text)
    embedder.embed(text=text)
    assert embedder.cache.size == len(text), "Cache should not be empty"

    embedder.clean()
    assert embedder.cache.size == 0, "Cache should be empty"

def test_embedder_tracker():
    embedder = BertEmbedder(pretrained_model_dir=PRETRAINED_DIR)
    tracker = Tracker()

    embedder.use_progress_tracker(tracker)
    assert embedder.get_progress_tracker() == tracker, "Tracker instances should match"

def test_embedder_components():
    embedder = BertEmbedder(pretrained_model_dir=PRETRAINED_DIR)
    cuda = embedder.get_cuda_components()
    infr = embedder.get_inferential_components()

    assert cuda == infr, "Components should match"

def test_embedder_params():
    embedder = BertEmbedder(pretrained_model_dir=PRETRAINED_DIR)

    assert not embedder.is_inference, "Should not be in inference mode"
    assert len( embedder.get_training_parameters()) > 0, "Parameters should not be empty"

    embedder.set_inference_mode(on = True)
    assert embedder.is_inference, "Shouild be in inference mode"
    assert len(embedder.get_training_parameters()) == 0, "Parameters should be empty"
