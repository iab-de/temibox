import pytest
import torch.nn

from temibox.context import Context
from temibox.model.classifier import BinaryClassifier
from temibox.losses import BinaryLoss
from temibox.pipeline import StandardPipeline
from temibox.tracker import Tracker

from _support import MockDocument, MockUseCase, MockLoss
from _support import MockTokenizer, MockVectorizer, MockEmbedder

#######################
# Setup
#######################

documents = [MockDocument(text="Das ist ein Test", label_id=0),
             MockDocument(text="Zweites Testdokument", label_id=1),
             MockDocument(text="Das letzte Testdokument", label_id=0)]

def get_default_model(multilabel: bool = False) -> BinaryClassifier:
    return BinaryClassifier(multilabel=multilabel,
                     layer_dimensions=[8, 4],
                     layer_activation=torch.nn.ReLU,
                     dropout=0.1,
                     loss_functions=[MockLoss()])

#######################
# Tests
#######################
def test_simple_train():
    model = get_default_model()

    with pytest.raises(Exception):
        model.train()

    model.train(embedder=MockEmbedder())
    model.train(embedder=MockEmbedder()) # Repeated train is fine

def test_multilabel():
    model = get_default_model(True)

    with pytest.raises(Exception):
        model.train()

    model.train(embedder=MockEmbedder())

def test_no_loss_functions():

    model = BinaryClassifier(multilabel=False,
                             layer_dimensions=[8,4],
                             layer_activation = torch.nn.ReLU,
                             dropout = 0.1)

    assert model._loss_functions is not None and len(model._loss_functions) > 0, "Loss functions missing"

    model = BinaryClassifier(multilabel=True,
                             layer_dimensions=[8,4],
                             layer_activation = torch.nn.ReLU,
                             dropout = 0.1)

    assert model._loss_functions is not None and len(model._loss_functions) > 0, "Loss functions missing"

def test_cuda_before_train():

    model = get_default_model()
    model.train(embedder=MockEmbedder())
    model.set_cuda_mode(on=True)

    assert model.is_cuda, "Model should be in cuda mode"
    assert model._model.parameters().__next__().device.type == "cuda", "Model should be on the GPU"

    model = get_default_model()

    model.set_cuda_mode(on = True)
    assert model.is_cuda, "Model should be in cuda mode"

    model.train(embedder=MockEmbedder())

    assert model.is_cuda, "Model should be in cuda mode"
    assert model._model.parameters().__next__().device.type == "cuda", "Model should be on the GPU"

def test_inference_before_train():

    model = get_default_model()

    model.train(embedder=MockEmbedder())
    model.set_inference_mode(on=True)

    assert model.is_inference, "Model should be in inference mode"
    assert not model._model.parameters().__next__().requires_grad, "Should be in eval mode"

    model = get_default_model()

    model.set_inference_mode(on = True)
    assert model.is_inference, "Model should be in inference mode"

    model.train(embedder=MockEmbedder())

    assert model.is_inference, "Model should be in inference mode"
    assert not model._model.parameters().__next__().requires_grad, "Should be in eval mode"

def test_forward():
    model = get_default_model()

    embedder = MockEmbedder()
    model.train(embedder = embedder)

    nr_docs = 4
    nr_labels = 64

    x = torch.rand((nr_docs, embedder.embedding_dim))
    targets = torch.rand((1, nr_labels, embedder.embedding_dim))
    y = model.forward(document_embeddings=x, target_embeddings=targets)

    assert y.shape == (4,64), "Prediction dimension is off (1)"

    y = model.forward(document_embeddings=x, target_embeddings=None)

    assert y.shape == (4, 1), "Prediction dimension is off (2)"

def test_set_identifier():

    model = get_default_model()

    assert model._identifier == "binary-classifier", "Standardidentifier falsch"
    model.use_identifier(new_identifier := "test-classifier")
    assert model._identifier == new_identifier, "Setzen des Identifiers fehlgeschlagen"

def test_use_loss_functions():
    model = BinaryClassifier(multilabel=False,
                             layer_dimensions=[8, 4],
                             layer_activation=torch.nn.ReLU,
                             dropout=0.1,
                             loss_functions=[BinaryLoss(use_class_weights=False)])

    assert len(model._loss_functions) == 1 and isinstance(model._loss_functions[0], BinaryLoss), "Standard loss function wrong"

    model.use_loss_functions([MockLoss()])
    assert len(model._loss_functions) == 1 and isinstance(model._loss_functions[0], MockLoss), "Setting loss function fails"

def test_get_loss():

    model = BinaryClassifier(multilabel=False,
                             layer_dimensions=[8, 4],
                             layer_activation=torch.nn.ReLU,
                             dropout=0.1,
                             loss_functions=[MockLoss()])

    usecase_1 = MockUseCase(name = "test-usecase")
    usecase_2 = MockUseCase(name = "test-usecase2")

    model.register_usecases([usecase_1])

    model.train(embedder = MockEmbedder())

    ctx = Context(None, [usecase_1, usecase_2], usecase_1)
    losses = model.get_losses(ctx, documents = documents)

    assert len(losses) == 1 and (loss := losses[0]).shape == (1,), "Loss shape is incorrect"
    assert loss.item() == len(documents), "Incorrect loss"

def test_tokenizer_vectorizer():
    model = get_default_model()

    with pytest.raises(Exception):
        model.train(tokenizer  = MockTokenizer(),
                    vectorizer = MockVectorizer(),
                    embedder   = MockEmbedder())

    model = get_default_model()

    model.train(tokenizer=MockTokenizer(),
                vectorizer=MockVectorizer())

def test_prepare_target_embeddings():
    usecase = MockUseCase()

    label_dict = {i:k for i, k in enumerate(usecase.get_usecase_labels())}

    model = get_default_model()
    model.train(embedder=MockEmbedder())

    model._prepare_target_embeddings(usecase_name = usecase.name,
                                     label_dict   = label_dict)

def test_predict():
    usecase = MockUseCase()

    label_dict = {i: k for i, k in enumerate(usecase.get_usecase_labels())}

    ctx = Context(None, [usecase], usecase)

    model = get_default_model()
    model.train(embedder=(embedder := MockEmbedder()))

    model._prepare_target_embeddings(usecase_name=usecase.name,
                                     label_dict=label_dict)

    model.register_usecases([usecase])

    text  = [usecase.get_document_body(d) for d in documents]
    embeddings = embedder.embed(text=text)
    preds = model.predict(ctx = ctx,
                          embeddings = embeddings,
                          documents = documents,
                          max_predictions = 10,
                          min_score = 0.0,
                          binary_threshold = 0.5)

    assert len(preds) == len(documents), "Number of predictions does not match"

def test_pipeline_predict():

    usecase    = MockUseCase()
    label_dict = {i: k for i, k in enumerate(usecase.get_usecase_labels())}
    model      = get_default_model()

    pipeline = StandardPipeline() \
        .add_usecase(usecase) \
        .add_step("embedder", MockEmbedder()) \
        .add_step("model", model, dependencies=["embedder"])

    pipeline.train(documents = documents)
    preds = pipeline.predict(documents  = documents,
                             label_dict = label_dict)

    assert len(preds) == len(documents), "Number of predictions does not match"

def test_predict_multilabel():
    usecase = MockUseCase()

    for test in [0, 1]:

        model = get_default_model(multilabel = True)
        model.train(embedder=(embedder := MockEmbedder()))
        model.register_usecases([usecase])

        label_dict = {i: k for i, k in enumerate(usecase.get_usecase_labels())}

        text  = [usecase.get_document_body(d) for d in documents]
        embeddings = embedder.embed(text=text)

        if test == 0:
            model.use_label_dict(usecase.name, label_dict)

        preds = model.predict(ctx = Context(None, [usecase], usecase),
                              embeddings = embeddings,
                              documents = documents,
                              label_dict = label_dict,
                              max_predictions = 10,
                              min_score = 0.0,
                              binary_threshold = 0.5)

        assert len(preds) == len(documents), "Number of predictions does not match"
        assert all([len(p.payload_raw) == len(usecase.get_usecase_labels()) for p in preds]), "Not all labels have predictions"

def test_tracker():
    model = get_default_model()
    tracker = Tracker()

    assert tracker != model.get_progress_tracker(), "Tracker instances should not match"
    model.use_progress_tracker(tracker)
    assert tracker == model.get_progress_tracker(), "Tracker instances should match"