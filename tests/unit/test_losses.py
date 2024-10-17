import torch
import numpy as np

from temibox.losses import BinaryLoss, MultilabelBinaryLoss, TripletLoss, MultinomialLoss
from temibox.context import Context
from temibox.model.classifier import BinaryClassifier, MultinomialClassifier

from _support import MockDocument, MockUseCase, MockEmbedder


#######################
# Setup
#######################


documents = [MockDocument(text="Dokument 1", label_id=0),
             MockDocument(text="Dokument 2", label_id=0),
             MockDocument(text="Dokument 3", label_id=0),
             MockDocument(text="Dokument 4", label_id=1),
             MockDocument(text="Dokument 5", label_id=1),
             MockDocument(text="Dokument 6", label_id=1)
]

mock_uc1 = MockUseCase()

def get_model(multilabel: bool) -> tuple[MockEmbedder, BinaryClassifier]:

    embedder = MockEmbedder()
    model = BinaryClassifier(layer_dimensions=[4], multilabel=multilabel)
    model.train(embedder=embedder)
    model.set_inference_mode(on=True)

    return embedder, model


def get_multi_model(multilabel: bool) -> tuple[MockEmbedder, MultinomialClassifier]:

    ctx = Context(None, [mock_uc1], mock_uc1)
    embedder = MockEmbedder()
    model = MultinomialClassifier(layer_dimensions=[32], multilabel=multilabel)
    model.register_usecases([mock_uc1])
    model.train(ctx = ctx, embedder=embedder)
    model.set_inference_mode(on=True)

    return embedder, model

#######################
# Tests
#######################

def test_binary_loss_no_weights():

    for _ in range(50):

        embedder, model = get_model(multilabel = False)

        bl = BinaryLoss(use_class_weights = False, scale = 1.0)

        result = bl(model=model, usecase=mock_uc1, embedder = embedder, documents=documents).item()

        label_ids = [d.label_id for d in documents]
        target_embeddings = embedder._last_embeddings[-1]

        y    = torch.FloatTensor(label_ids)
        yhat = model.forward(target_embeddings)

        bce = -1/len(y) * sum([y[i]*np.log(yhat[i]) + (1-y[i])*np.log(1-yhat[i]) for i in range(len(y))])

        assert abs(bce - result) < 1e-4, "Invalid loss result"


def test_binary_loss_weights():

    for _ in range(50):

        embedder, model = get_model(multilabel=False)

        bl = BinaryLoss(use_class_weights=True, scale=1.0)

        result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        label_ids = [d.label_id for d in documents]
        target_embeddings = embedder._last_embeddings[-1]

        y = torch.FloatTensor(label_ids)
        yhat = model.forward(target_embeddings)
        weights = mock_uc1.get_label_inverse_weights([["a", "b"][lid] for lid in label_ids])

        bce = -1 / len(y) * sum([weights[i] * (y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i])) for i in range(len(y))])

        assert abs(bce - result) < 1e-4, "Invalid loss result"

def test_multilabel_loss_no_weights():

    for _ in range(50):

        embedder, model = get_model(multilabel=False)

        bl = MultilabelBinaryLoss(positive_examples = 2, negative_examples = 2, use_class_weights=False, scale=1.0)
        result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        document_embeddings = embedder._last_embeddings[0]
        target_embeddings   =  torch.concat([emb.unsqueeze(0) for emb in embedder._last_embeddings[1:]])

        test_labels_idx = [[1]*2 + [0]*2 for _ in documents]
        y = torch.concat([torch.FloatTensor(ti).unsqueeze(0) for ti in test_labels_idx])
        yhat = model.forward(document_embeddings, target_embeddings)

        bce = (-1 / len(y) * sum([y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i]) for i in range(len(y))])).mean()

    assert abs(bce - result) < 1e-4, "Invalid loss result"

def test_multilabel_loss_weights():

    for _ in range(50):

        embedder, model = get_model(multilabel=False)

        bl = MultilabelBinaryLoss(positive_examples = 2, negative_examples = 2, use_class_weights=True, scale=1.0)
        result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        document_embeddings = embedder._last_embeddings[0]
        target_embeddings   =  torch.concat([emb.unsqueeze(0) for emb in embedder._last_embeddings[1:]])

        test_labels     = [mock_uc1.create_document_test_labels(d, 2, 2) for d in documents]
        test_labels_str = [[x[0] for x in d] for d in test_labels]
        test_labels_idx = [[int(x[1]) for x in d] for d in test_labels]

        y = torch.concat([torch.FloatTensor(ti).unsqueeze(0) for ti in test_labels_idx])
        yhat = model.forward(document_embeddings, target_embeddings)

        weights = torch.FloatTensor(mock_uc1.get_label_inverse_weights([x for d in test_labels_str for x in d])).reshape(len(documents),-1)

        bce = (-1 / len(y) * sum([weights[i] * (y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i])) for i in range(len(y))])).mean()

        assert abs(bce - result) < 1e-4, "Invalid loss result"


def test_multinomial_loss_no_weights():

    for _ in range(50):

        embedder, model = get_multi_model(multilabel=False)

        bl = MultinomialLoss(use_class_weights=False, scale=1.0, positive_weight=1)
        result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        document_embeddings = embedder._last_embeddings[0]

        y = torch.concat([torch.FloatTensor(mock_uc1.get_multinomial_indicators(d)).unsqueeze(0) for d in documents])
        yhat = model.forward(mock_uc1.name, document_embeddings)

        bce = (-1 / len(y) * sum([y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i]) for i in range(len(y))])).mean()

        assert abs(bce - result) < 1e-4, "Invalid loss result"

def test_multinomial_loss_no_weights_pos_w():

    for _ in range(50):

        for pos_w in [2, 5, 10, 100]:
            embedder, model = get_multi_model(multilabel=False)

            bl = MultinomialLoss(use_class_weights=False, scale=1.0, positive_weight=pos_w)
            result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

            document_embeddings = embedder._last_embeddings[0]

            y = torch.concat([torch.FloatTensor(mock_uc1.get_multinomial_indicators(d)).unsqueeze(0) for d in documents])
            yhat = model.forward(mock_uc1.name, document_embeddings)

            weights = y * min(pos_w, len(mock_uc1.get_usecase_labels())) + 1

            bce = (-1 / len(y) * sum([weights[i] * (y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i])) for i in range(len(y))])).mean()

            assert abs(bce - result) < 1e-4, "Invalid loss result"

def test_multinomial_loss_weights():

    for _ in range(50):

        embedder, model = get_multi_model(multilabel=False)

        bl = MultinomialLoss(use_class_weights=True, scale=1.0)
        result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        document_embeddings = embedder._last_embeddings[0]

        y = torch.concat([torch.FloatTensor(mock_uc1.get_multinomial_indicators(d)).unsqueeze(0) for d in documents])
        yhat = model.forward(mock_uc1.name, document_embeddings)

        weights = torch.FloatTensor(list(mock_uc1.get_usecase_label_inverse_weights().values())).repeat(len(documents),1)

        bce = (-1 / len(y) * sum([weights[i] * (y[i] * np.log(yhat[i]) + (1 - y[i]) * np.log(1 - yhat[i])) for i in range(len(y))])).mean()

        assert abs(bce - result) < 1e-4, "Invalid loss result"


def test_triplet_loss_no_weights():

    for examples in range(1, 10):

        embedder, model = get_model(multilabel=False)
        bl = TripletLoss(use_class_weights=False, examples=examples, scale=1.0)
        r_result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        pos_emb, anc_emb, neg_emb = embedder._last_embeddings[-3:]

        d_ap = torch.norm(anc_emb - pos_emb, p=2, dim=1)
        d_an = torch.norm(anc_emb - neg_emb, p=2, dim=1)

        c_result = (torch.maximum(d_ap - d_an + 1, torch.zeros_like(d_ap)).sum()/len(documents)).item()

        assert abs(c_result - r_result) < 1e-3, "Invalid loss result"


def test_triplet_loss_weights():

    for examples in range(1,10):

        embedder, model = get_model(multilabel=False)
        bl = TripletLoss(use_class_weights=True, examples=examples, scale=1.0)
        r_result = bl(model=model, usecase=mock_uc1, embedder=embedder, documents=documents).item()

        pos_emb, anc_emb, neg_emb = embedder._last_embeddings[-3:]

        d_ap = torch.norm(anc_emb - pos_emb, p=2, dim=1)
        d_an = torch.norm(anc_emb - neg_emb, p=2, dim=1)

        loss =  torch.maximum(d_ap - d_an + 1, torch.zeros_like(d_ap)).unsqueeze(1)
        weights = torch.FloatTensor([w for d in documents for w in examples * [sum(mock_uc1.get_label_inverse_weights(mock_uc1.get_document_labels(d)))]]).unsqueeze(0)

        c_result = (torch.mm(weights, loss).squeeze(0)/weights.sum().squeeze(0)).item()

        assert abs(c_result - r_result) < 1e-3, "Invalid loss result"

def test_triplet_no_triplets():

    embedder, model = get_model(multilabel=False)
    bl = TripletLoss(use_class_weights=True, examples=1, scale=1.0)
    r_result = bl(model=model, usecase=MockUseCase(no_triplets=True), embedder=embedder, documents=documents).item()

    assert abs(r_result) < 1e-3, "Should be zero"
