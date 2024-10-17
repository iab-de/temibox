import pytest
import torch.nn

from temibox.context import Context
from temibox.tracker import Tracker
from temibox.model.classifier import MultinomialClassifier
from temibox.losses import MultinomialLoss

from _support import MockMultiUseCase, MockDocumentMulti, MockTokenizer, MockVectorizer, MockEmbedder, MockLoss

#######################
# Setup
#######################

multi_documents = [
                   MockDocumentMulti(text="Dokument 1", label_ids=[1,2,3]),
                   MockDocumentMulti(text="Dokument 2", label_ids=[1,2]),
                   MockDocumentMulti(text="Dokument 3", label_ids=[1,3]),
                   MockDocumentMulti(text="Dokument 4", label_ids=[2,3]),
                   MockDocumentMulti(text="Dokument 5", label_ids=[2])]

labels_1 = {"a": 0.1, "b": 0.7, "c": 0.2}
labels_2 = {"d": 0.1, "e": 0.2, "f": 0.3, "g": 0.3, "h": 0.1}

label_dict_1 = {123: "a",
                225: "b",
                227: "c"}

label_dict_2 = {1020: "d",
                1021: "e",
                1022: "f",
                1023: "g",
                1024: "h"}

label_combo_1 = {v: (k, labels_1[v]) for k,v in label_dict_1.items()}
label_combo_2 = {v: (k, labels_2[v]) for k,v in label_dict_2.items()}

#######################
# Tests
#######################

def test_train():
    usecase_1 = MockMultiUseCase(name = "usecase_1", use_labels = label_combo_1)
    usecase_2 = MockMultiUseCase(name = "usecase_2", use_labels = label_combo_2)

    embedder = MockEmbedder()

    for multilabel in [True, False]:
        model = MultinomialClassifier(multilabel = multilabel,
                                      layer_dimensions = [256, 32])

        model.register_usecases([usecase_1, usecase_2])
        ctx = Context(None, [usecase_1, usecase_2], usecase_1)

        model.train(ctx = ctx, embedder = embedder)
        model.train(ctx = ctx, embedder = embedder) # Repeated train is OK

        assert len(model._heads) == 2, "Should have two heads"
        assert model._heads.keys() == {usecase_1.name, usecase_2.name}, "Should have a head for each usecase"
        assert model._heads[usecase_1.name].output.out_features == len(usecase_1.get_usecase_labels()), "Output dimensions wrong (1)"
        assert model._heads[usecase_2.name].output.out_features == len(usecase_2.get_usecase_labels()), "Output dimensions wrong (2)"

        if multilabel:
            assert type(model._heads[usecase_1.name].output_activation) == torch.nn.Sigmoid, "Output activation should be a Sigmoid (1)"
            assert type(model._heads[usecase_2.name].output_activation) == torch.nn.Sigmoid, "Output activation should be a Sigmoid (2)"
        else:
            assert type(model._heads[usecase_1.name].output_activation) == torch.nn.Softmax, "Output activation should be a Softmax (1)"
            assert type(model._heads[usecase_2.name].output_activation) == torch.nn.Softmax, "Output activation should be a Softmax (2)"

def test_train_predict():
    usecase_1 = MockMultiUseCase(name = "usecase_1", use_labels = label_combo_1.copy())
    usecase_2 = MockMultiUseCase(name = "usecase_2", use_labels = label_combo_2.copy())

    embedder = MockEmbedder()

    for multilabel in [True, False]:
        model = MultinomialClassifier(multilabel = multilabel,
                                      layer_dimensions = [256, 32])

        model.register_usecases([usecase_1, usecase_2])
        ctx_1 = Context(None, [usecase_1, usecase_2], usecase_1)
        ctx_2 = Context(None, [usecase_1, usecase_2], usecase_2)

        model.train(ctx = ctx_1, embedder = embedder)

        texts = ["Unit Test", "Unit Test2"]
        embeddings = embedder.embed(text=texts)

        for ctx, label_dict in [(ctx_1, label_dict_1), (ctx_2, label_dict_2)]:

            preds = model.predict(ctx = ctx,
                                  label_dict=label_dict.copy(),
                                  embeddings = embeddings)

            assert len(preds) == len(texts), "Number of predictions wrong"
            for pred in preds:
                assert len(pred.payload_raw) == len(label_dict), "Wrong number of raw predictions"
                assert label_dict.keys() & {p.label_id for p in pred.payload_raw} == label_dict.keys(), "Not all label IDs present"
                assert set(label_dict.values()) & {p.label for p in pred.payload_raw} == set(label_dict.values()), "Not all label values present"


def test_train_tok_vec_nok():
    usecase_1 = MockMultiUseCase(name = "usecase_1", use_labels = label_combo_1)
    usecase_2 = MockMultiUseCase(name = "usecase_2", use_labels = label_combo_2)

    embedder = MockEmbedder()
    tokenizer = MockTokenizer()
    vectorizer = MockVectorizer()

    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[256, 32])

    model.register_usecases([usecase_1, usecase_2])
    ctx = Context(None, [usecase_1, usecase_2], usecase_1)

    with pytest.raises(Exception):
        model.train(ctx)

    with pytest.raises(Exception):
        model.train(ctx,
                    tokenizer=tokenizer)

    with pytest.raises(Exception):
        model.train(ctx,
                    vectorizer=vectorizer)

    with pytest.raises(Exception):
        model.train(ctx,
                    embedder = embedder,
                    tokenizer = tokenizer,
                    vectorizer = vectorizer)


def test_train_tok_vec_ok():
    usecase_1 = MockMultiUseCase(name = "usecase_1", use_labels = label_combo_1)
    usecase_2 = MockMultiUseCase(name = "usecase_2", use_labels = label_combo_2)

    embedder = MockEmbedder()
    tokenizer = MockTokenizer()
    vectorizer = MockVectorizer()

    texts = ["Unit Test", "Unit Test2"]
    embeddings = embedder.embed(text=texts)
    label_dict = label_dict_1.copy()

    for kwargs in [{"embedder": embedder}, {"tokenizer": tokenizer, "vectorizer": vectorizer}]:
        model = MultinomialClassifier(multilabel=True,
                                      dropout = 0.1,
                                      layer_dimensions=[256, 32])

        model.register_usecases([usecase_1, usecase_2])
        ctx = Context(None, [usecase_1, usecase_2], usecase_1)
        model.train(ctx=ctx, **kwargs)

        assert len(model._heads) == 2, "Should have two heads"
        assert model._heads.keys() == {usecase_1.name, usecase_2.name}, "Should have a head for each usecase"
        assert model._heads[usecase_1.name].output.out_features == len(usecase_1.get_usecase_labels()), "Output dimensions wrong (1)"
        assert model._heads[usecase_2.name].output.out_features == len(usecase_2.get_usecase_labels()), "Output dimensions wrong (2)"
        assert type(model._heads[usecase_1.name].output_activation) == torch.nn.Sigmoid, "Output activation should be a Sigmoid (1)"
        assert type(model._heads[usecase_2.name].output_activation) == torch.nn.Sigmoid, "Output activation should be a Sigmoid (2)"

        preds = model.predict(ctx=ctx,
                              label_dict=label_dict.copy(),
                              embeddings=embeddings)

        assert len(preds) == len(texts), "Number of predictions wrong"
        for pred in preds:
            assert len(pred.payload_raw) == len(label_dict), "Wrong number of raw predictions"
            assert label_dict.keys() & {p.label_id for p in pred.payload_raw} == label_dict.keys(), "Not all label IDs present"
            assert set(label_dict.values()) & {p.label for p in pred.payload_raw} == set(label_dict.values()), "Not all label values present"

def test_cuda_and_inference():
    usecase = MockMultiUseCase()
    embedder = MockEmbedder()


    for cuda_before in [True, False]:
        model = MultinomialClassifier(multilabel = True,
                                      layer_dimensions = [256, 32])

        model.register_usecases([usecase])
        ctx = Context(None, [usecase], usecase)

        assert not model.is_cuda, "Should not be in cuda mode"
        assert not model.is_inference, "Should not be in inference mode"
        assert len(model.get_training_parameters()) == 0, "Should not have training params"

        if cuda_before:
            model.set_cuda_mode(on = True)
            model.set_inference_mode(on = True)
            assert model.is_cuda, "Should be in cuda mode"
            assert model.is_inference, "Should be in inference mode"

            model.train(ctx = ctx, embedder = embedder)
        else:
            model.train(ctx=ctx, embedder=embedder)
            model.set_cuda_mode(on=True)
            model.set_inference_mode(on=True)
            assert model.is_cuda, "Should be in cuda mode"
            assert model.is_inference, "Should be in inference mode"

        model.set_inference_mode(on=False)
        assert len(model.get_training_parameters()) > 0, "Should have training params"

        model.set_inference_mode(on = True)
        assert model.is_inference, "Should be in inference mode"
        assert len(model.get_training_parameters()) == 0, "Should not have training params"

def test_training_params():

    usecase_1 = MockMultiUseCase(name="usecase_1", use_labels=label_combo_1)
    usecase_2 = MockMultiUseCase(name="usecase_2", use_labels=label_combo_2)

    embedder = MockEmbedder()

    for layer_dimensions in [[32], [256, 32], [256, 64, 32, 8]]:
        for usecases in [[usecase_1, usecase_2], [usecase_1], [usecase_2]]:
            model = MultinomialClassifier(multilabel=True,
                                          layer_dimensions=layer_dimensions)

            model.register_usecases(usecases)
            ctx = Context(None, usecases, usecases[-1])

            model.train(ctx=ctx, embedder=embedder)

            assert len(model._heads) == len(usecases), "Should have two heads"
            assert not model.is_inference, "Should not be in inference mode"

            params = model.get_training_parameters()
            assert len(params) == len(layer_dimensions) * 2 + len(usecases)*2, "Wrong parameter list"

            model.set_inference_mode(on = True)
            params = model.get_training_parameters()
            assert len(params) == 0, "Should not have any parameters"

def test_forward():
    usecase = MockMultiUseCase()

    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1])

    model.register_usecases([usecase])
    ctx = Context(None, [usecase], usecase)

    embedder = MockEmbedder()
    model.train(ctx = ctx, embedder = embedder)

    nr_docs = 23
    nr_labels = len(usecase.get_usecase_labels())

    x = torch.rand((nr_docs, embedder.embedding_dim))
    y = model.forward(usecase_name=usecase.name, document_embeddings=x)

    assert y.shape == (nr_docs, nr_labels), "Prediction dimension is off (1)"

def test_forward_softmax():
    usecase = MockMultiUseCase()

    model = MultinomialClassifier(multilabel=False,
                                  layer_dimensions=[32])

    model.register_usecases([usecase])
    ctx = Context(None, [usecase], usecase)

    embedder = MockEmbedder()
    model.train(ctx = ctx, embedder = embedder)

    nr_docs = 23
    nr_labels = len(usecase.get_usecase_labels())

    x = torch.rand((nr_docs, embedder.embedding_dim))
    y = model.forward(usecase_name=usecase.name, document_embeddings=x)

    assert y.shape == (nr_docs, nr_labels), "Prediction dimension is off (1)"
    assert torch.allclose(y.sum(dim=1), torch.ones(nr_docs)), "Row-wise sum should be equal to 1"

def test_use_loss_functions():

    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1])

    assert len(model._loss_functions) == 1 and isinstance(model._loss_functions[0], MultinomialLoss), "Standard loss function wrong"
    assert model._loss_functions[0]._use_class_weights == False, "standard use_class_weights is Wrong"
    assert abs(model._loss_functions[0]._scale - 1.0) < 1e-5, "Standard scale is wrong"

    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1],
                                  loss_functions=[MultinomialLoss(use_class_weights=True, scale=2.3)])

    assert len(model._loss_functions) == 1 and isinstance(model._loss_functions[0], MultinomialLoss), "Chosen loss function wrong"
    assert model._loss_functions[0]._use_class_weights == True, "use_class_weights is Wrong"
    assert abs(model._loss_functions[0]._scale - 2.3) < 1e-5, "Scale is wrong"

    model.use_loss_functions([MockLoss()])
    assert len(model._loss_functions) == 1 and isinstance(model._loss_functions[0], MockLoss), "Setting loss function fails"


def test_get_loss():

    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1],
                                  loss_functions=[MockLoss()])

    usecase_1 = MockMultiUseCase(name = "test-usecase")
    usecase_2 = MockMultiUseCase(name = "test-usecase2")

    ctx = Context(None, [usecase_1, usecase_2], usecase_1)
    model.register_usecases([usecase_1])

    model.train(ctx = ctx, embedder = MockEmbedder())

    losses = model.get_losses(ctx, documents = multi_documents)

    assert len(losses) == 1 and (loss := losses[0]).shape == (1,), "Loss shape is incorrect"
    assert loss.item() == len(multi_documents), "Incorrect loss"

def test_tracker():
    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1])

    tracker = Tracker()

    assert tracker != model.get_progress_tracker(), "Tracker instances should not match"
    model.use_progress_tracker(tracker)
    assert tracker == model.get_progress_tracker(), "Tracker instances should match"


def test_set_identifier():
    model = MultinomialClassifier(multilabel=True,
                                  layer_dimensions=[1])

    assert model._identifier == "multinomial-classifier", "Standardidentifier falsch"
    model.use_identifier(new_identifier := "test-classifier")
    assert model._identifier == new_identifier, "Setzen des Identifiers fehlgeschlagen"