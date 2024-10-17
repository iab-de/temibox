import pytest
import numpy as np
from pandas import DataFrame

from temibox.evaluation import Evaluator
from temibox.pipeline import StandardPipeline
from temibox.context import ContextArg
from temibox.prediction import RawPrediction
from temibox.evaluation.metric import Accuracy, F1, ConfusionMatrix, ScoreHistogram, PrecisionAtK, RecallAtK, Calibration, RocAuc

from _support import MockUseCase, MockMultiUseCase
from _support import MockDocument, MockDocumentMulti, MockMetric
from _support import TestTrainablePredictable

#######################
# Setup
#######################

documents = [MockDocument(text="Dokument 1", label_id=0),
             MockDocument(text="Dokument 2", label_id=0),
             MockDocument(text="Dokument 3", label_id=0),
             MockDocument(text="Dokument 4", label_id=1),
             MockDocument(text="Dokument 5", label_id=1),
             MockDocument(text="Dokument 6", label_id=1)]

raw_preds = [[RawPrediction(label="a", label_id=0, score=0.25)],
             [RawPrediction(label="b", label_id=1, score=0.75)],
             [RawPrediction(label="a", label_id=0, score=0.05)],
             [RawPrediction(label="b", label_id=1, score=0.95)],
             [RawPrediction(label="b", label_id=1, score=0.65)],
             [RawPrediction(label="b", label_id=1, score=0.75)]]

r_p = sum([d.label_id == 1 for d in documents])
r_n = sum([d.label_id == 0 for d in documents])
p_p = sum([p[0].label_id == 1 for p in raw_preds])
p_n = sum([p[0].label_id == 0 for p in raw_preds])

tp = sum([(d.label_id == 1 and p[0].label_id == 1) for d, p in zip(documents, raw_preds)])
fp = sum([(d.label_id == 0 and p[0].label_id == 1) for d, p in zip(documents, raw_preds)])
tn = sum([(d.label_id == 0 and p[0].label_id == 0) for d, p in zip(documents, raw_preds)])
fn = sum([(d.label_id == 1 and p[0].label_id == 0) for d, p in zip(documents, raw_preds)])

multi_documents = [
                   MockDocumentMulti(text="Dokument 1", label_ids=[1,2,3]),
                   MockDocumentMulti(text="Dokument 2", label_ids=[1,2]),
                   MockDocumentMulti(text="Dokument 3", label_ids=[1,3]),
                   MockDocumentMulti(text="Dokument 4", label_ids=[2,3]),
                   MockDocumentMulti(text="Dokument 5", label_ids=[2])]

multi_raw_preds = [
                    [RawPrediction(label="a", label_id=1, score=0.75),
                     RawPrediction(label="b", label_id=2, score=0.85),
                     RawPrediction(label="c", label_id=3, score=0.95)],

                    [RawPrediction(label="a", label_id=1, score=0.75),
                     RawPrediction(label="b", label_id=2, score=0.75),
                     RawPrediction(label="c", label_id=3, score=0.25)],

                    [RawPrediction(label="a", label_id=1, score=0.95),
                     RawPrediction(label="b", label_id=2, score=0.85),
                     RawPrediction(label="c", label_id=3, score=0.75)],

                    [RawPrediction(label="a", label_id=1, score=0.05),
                     RawPrediction(label="b", label_id=2, score=0.05),
                     RawPrediction(label="c", label_id=3, score=0.15)],

                    [RawPrediction(label="a", label_id=1, score=0.05),
                     RawPrediction(label="b", label_id=2, score=0.45),
                     RawPrediction(label="c", label_id=3, score=0.05)]
                   ]

mock_uc_1 = MockUseCase(name = "usecase-1")

def get_multi_matches(k: int,
                      multi_documents: list[MockDocumentMulti],
                      multi_raw_preds: list[list[RawPrediction]],
                      threshold: float) -> list[tuple[int, int]]:

    result = [None] * len(multi_documents)
    for i, (d, p) in enumerate(zip(multi_documents, multi_raw_preds)):

        preds   = {pi.label_id for pi in sorted(p, key = lambda x: x.score, reverse=True)[:k] if pi.score >= threshold}
        real    = set(d.label_ids)
        matches = real & preds

        result[i] = (real, matches, preds)

    return result

def get_roc_auc(documents, raw_preds):

    step_count = 10000
    tpr = [0] * step_count
    fpr = [0] * step_count
    for step in range(0, step_count):
        threshold = step/step_count

        tp = sum([(d.label_id == 1 and int(p[0].score >= threshold) == 1) for d, p in zip(documents, raw_preds)])
        fp = sum([(d.label_id == 0 and int(p[0].score >= threshold) == 1) for d, p in zip(documents, raw_preds)])
        tn = sum([(d.label_id == 0 and int(p[0].score >= threshold) == 0) for d, p in zip(documents, raw_preds)])
        fn = sum([(d.label_id == 1 and int(p[0].score >= threshold) == 0) for d, p in zip(documents, raw_preds)])

        tpr[step] = tp / (tp + fn)
        fpr[step] = fp / (fp + tn)

    return sum([(fpr[i-1]-fpr[i]) * (tpr[i-1] + tpr[i])/2 for i in range(1, step_count)])


#######################
# Tests
#######################

def test_simple_evaluation():

    pipeline = StandardPipeline() \
        .add_usecase(MockUseCase()) \
        .add_step("test-model", TestTrainablePredictable(name = "test-model", activities = (activities := []))) \
        .add_step("evaluator", Evaluator(metrics = [MockMetric()]))

    pipeline.train()
    pipeline.predict()

    activities.clear()
    out = pipeline.evaluate(documents = documents,
                            reps=len(documents),
                            transformed_value = (tv := 12))

    assert activities == ["test-model-predict"], "Activities are incorrect"
    assert "evaluator" in out, "Evaluation results were not returned"
    metrics = ContextArg.extract(out["evaluator"]["metrics"])
    assert "test-model" in metrics, "test-model results are not contained in the evaluation results"
    assert "mock-metric" in metrics["test-model"], "Mock-metric results not present"
    assert metrics["test-model"]["mock-metric"] == tv*3, "Metric value incorrect"


def test_evaluate_before_training():

    pipeline = StandardPipeline() \
        .add_usecase(MockUseCase()) \
        .add_step("test-model", TestTrainablePredictable(name = "test-model", activities = [])) \
        .add_step("evaluator", Evaluator(metrics = [MockMetric()]))

    out = pipeline.evaluate(documents = documents,
                            reps=len(documents))

    assert len(out) == 0, "Evaluation should be empty"


def test_irrelevant_usecases():

    pipeline = StandardPipeline() \
        .add_usecase(uc_1 := MockUseCase(name = "usecase-1")) \
        .add_usecase(uc_2 := MockUseCase(name = "usecase-2")) \
        .add_step("test-model", TestTrainablePredictable(name = "test-model", activities = (activities := []))) \
        .add_step("evaluator", Evaluator(metrics = [MockMetric()]), usecases=[uc_2])

    pipeline.train()
    out = pipeline.evaluate(documents = documents, reps=len(documents))

    metrics_1 = ContextArg.extract(out["evaluator"]["metrics"], usecase_name=uc_1.name)
    metrics_2 = ContextArg.extract(out["evaluator"]["metrics"], usecase_name=uc_2.name)

    assert metrics_1 is None, "Usecase-1 will not be evaluated"
    assert metrics_2 is not None and len(metrics_2) > 0, "Usecase-2 should be evaluated"


def test_preds_vs_docs():

    pipeline = StandardPipeline() \
        .add_usecase(MockUseCase()) \
        .add_step("test-model", TestTrainablePredictable(name = "test-model", activities = (activities := []))) \
        .add_step("evaluator", Evaluator(metrics = [MockMetric()]))

    pipeline.train()

    assert len(documents) > 1, "This test requires more than one document"

    with pytest.raises(Exception):
        pipeline.evaluate()

    with pytest.raises(Exception):
        pipeline.evaluate(documents = documents, reps=1)

def test_replace_metrics():

    metric_1 = MockMetric(name="metric-1")
    metric_2 = MockMetric(name="metric-2")
    metric_3 = MockMetric(name="metric-3")

    evaluator = Evaluator(metrics = [metric_1, metric_3])

    assert len(evaluator._metrics) == 2, "Evaluator should have two metrics"
    assert evaluator._metrics[0] == metric_1, "First metric incorrect"
    assert evaluator._metrics[1] == metric_3, "Second metric incorrect"

    evaluator.use_metrics([metric_2])
    assert len(evaluator._metrics) == 1, "Evaluator should have two metrics"
    assert evaluator._metrics[0] == metric_2, "Metric incorrect"

def test_accuracy():
    m_acc = Accuracy()
    assert m_acc.name == "accuracy", "Wrong metric name"

    result = m_acc(raw_predictions = raw_preds, usecase = mock_uc_1, documents = documents)

    assert abs(result["global"] - 5 / 6) < 1e-5, "Global accuracy is wrong"
    assert abs(result["per_label"]["a"] - 2 / 2) < 1e-5, "Label a accuracy is wrong"
    assert abs(result["per_label"]["b"] - 3 / 4) < 1e-5, "Label b accuracy is wrong"

def test_accuracy_df():
    m_acc = Accuracy()
    assert m_acc.name == "accuracy", "Wrong metric name"

    result_df = m_acc(raw_predictions = raw_preds, usecase = mock_uc_1, documents = documents, return_dataframe=True)

    assert result_df.shape == (2,3), "Invalid df shape"

    assert abs(result_df["overall"][0] - 5 / 6) < 1e-5, "Global accuracy is wrong"
    assert abs(result_df["accuracy"][0] - 2 / 2) < 1e-5, "Label a accuracy is wrong"
    assert abs(result_df["accuracy"][1] - 3 / 4) < 1e-5, "Label b accuracy is wrong"

def test_f1():
    m_f1 = F1()
    assert m_f1.name == "f1", "Wrong metric name"

    result = m_f1(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents)

    r_precision, r_recall, r_f1 = [float(x.split()[1].strip("%"))/100 for x in result.split("\n")]

    c_precision = tp/(tp+fp)
    c_recall    = tp/(tp+fn)
    c_f1 = 2*(c_precision*c_recall)/(c_precision+c_recall)

    assert abs(r_precision - c_precision) < 1e-4, "Precision is wrong"
    assert abs(r_recall - c_recall) < 1e-4, "Recall is wrong"
    assert abs(r_f1 - c_f1) < 1e-4, "F1 is wrong"

def test_f1_df():
    m_f1 = F1()
    assert m_f1.name == "f1", "Wrong metric name"

    result_df = m_f1(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents, return_dataframe=True)

    r_precision, r_recall, r_f1 = result_df.loc[0,:]

    c_precision = tp/(tp+fp)
    c_recall    = tp/(tp+fn)
    c_f1 = 2*(c_precision*c_recall)/(c_precision+c_recall)

    assert abs(r_precision - c_precision) < 1e-4, "Precision is wrong"
    assert abs(r_recall - c_recall) < 1e-4, "Recall is wrong"
    assert abs(r_f1 - c_f1) < 1e-4, "F1 is wrong"

def test_calibration():
    m_c = Calibration()
    assert m_c.name == "calibration", "Wrong metric name"

    result = m_c(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents)

    assert abs(float(result.loc["a", "real"].strip("%")) / 100 - r_n / (r_p + r_n)) < 1e-4, "Proportion for label a (real) is wrong"
    assert abs(float(result.loc["b", "real"].strip("%")) / 100 - r_p / (r_p + r_n)) < 1e-4, "Proportion for label b (real) is wrong"
    assert abs(float(result.loc["a", "pred"].strip("%")) / 100 - p_n / (r_p + r_n)) < 1e-4, "Proportion for label a (pred) is wrong"
    assert abs(float(result.loc["b", "pred"].strip("%")) / 100 - p_p / (r_p + r_n)) < 1e-4, "Proportion for label b (pred) is wrong"

def test_confusion_matrix():
    m_c = ConfusionMatrix()
    assert m_c.name == "confusion matrix", "Wrong metric name"

    result = m_c(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents)

    assert int(result.loc["b", "TN"]) == tn, "Cell TN is wrong"
    assert int(result.loc["b", "TP"]) == tp, "Cell TP is wrong"
    assert int(result.loc["b", "FP"]) == fp, "Cell FP is wrong"
    assert int(result.loc["b", "FN"]) == fn, "Cell FN is wrong"

    c_precision = tp / (tp + fp)
    c_recall = tp / (tp + fn)
    c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)

    assert abs(result.loc["b", "precision"] - c_precision) < 1e-5, "Cell precision is wrong"
    assert abs(result.loc["b", "recall"] - c_recall) < 1e-5, "Cell recall is wrong"
    assert abs(result.loc["b", "f1"] - c_f1) < 1e-5, "Cell f1 is wrong"

def test_confusion_matrix_pct():
    m_c = ConfusionMatrix(show_percent = True)
    assert m_c.name == "confusion matrix (%)", "Wrong metric name"

    result = m_c(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents)

    assert abs(float(result.loc["b", "TN"].strip("%")) / 100 - tn / (r_p + r_n)) < 1e-4, "Cell TN% is wrong"
    assert abs(float(result.loc["b", "TP"].strip("%")) / 100 - tp / (r_p + r_n)) < 1e-4, "Cell TP% is wrong"
    assert abs(float(result.loc["b", "FP"].strip("%")) / 100 - fp / (r_p + r_n)) < 1e-4, "Cell FP% is wrong"
    assert abs(float(result.loc["b", "FN"].strip("%")) / 100 - fn / (r_p + r_n)) < 1e-4, "Cell FN% is wrong"

    c_precision = tp / (tp + fp)
    c_recall = tp / (tp + fn)
    c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)

    assert abs(result.loc["b", "precision"] - c_precision) < 1e-5, "Cell precision is wrong"
    assert abs(result.loc["b", "recall"] - c_recall) < 1e-5, "Cell recall is wrong"
    assert abs(result.loc["b", "f1"] - c_f1) < 1e-5, "Cell f1 is wrong"

def test_precision_at_k():

    m_p = PrecisionAtK(k=3, min_score=0.0)
    with pytest.raises(Exception):
        m_p(raw_predictions=multi_raw_preds, usecase=MockMultiUseCase(), documents=multi_documents[1:])

    for k, score in [(10, 0.5), (10, 0.0),
                     (3, 0.5), (3, 0.0),
                     (2, 0.5), (2, 0.0), (2, 0.9),
                     (1, 0.5), (1, 0.0), (1, 0.9)]:

        m_p = PrecisionAtK(k=k, min_score=score)
        assert m_p.name == f"precision@k (k={k}, min-score={score:.2f})", "Wrong metric name"

        c_result = m_p(raw_predictions = multi_raw_preds, usecase = MockMultiUseCase(), documents = multi_documents)
        r_result = get_multi_matches(k, multi_documents, multi_raw_preds, threshold=score)

        total = lambda r: max(1, min(len(r[2]), k))

        t_mean = np.mean([len(r[1])/total(r) for r in r_result])
        t_1 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 1])
        t_2 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 2])
        t_3 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 3])

        assert (c_result["mean"] - t_mean) < 1e-4, "Global precision wrong"
        assert (c_result["mean_per_true_count"][1] - t_1) < 1e-4, "Count == 1 precision wrong"
        assert (c_result["mean_per_true_count"][2] - t_2) < 1e-4, "Count == 2 precision wrong"
        assert (c_result["mean_per_true_count"][3] - t_3) < 1e-4, "Count == 3 precision wrong"


def test_precision_at_k_df():

    m_p = PrecisionAtK(k=3, min_score=0.0)
    with pytest.raises(Exception):
        m_p(raw_predictions=multi_raw_preds, usecase=MockMultiUseCase(), documents=multi_documents[1:])

    for k, score in [(10, 0.5), (10, 0.0),
                     (3, 0.5), (3, 0.0),
                     (2, 0.5), (2, 0.0), (2, 0.9),
                     (1, 0.5), (1, 0.0), (1, 0.9)]:

        m_p = PrecisionAtK(k=k, min_score=score)
        assert m_p.name == f"precision@k (k={k}, min-score={score:.2f})", "Wrong metric name"

        c_result_df = m_p(raw_predictions = multi_raw_preds, usecase = MockMultiUseCase(), documents = multi_documents, return_dataframe=True)
        r_result = get_multi_matches(k, multi_documents, multi_raw_preds, threshold=score)

        total = lambda r: max(1, min(len(r[2]), k))

        t_mean = np.mean([len(r[1])/total(r) for r in r_result])
        t_1 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 1])
        t_2 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 2])
        t_3 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 3])

        assert (c_result_df["overall"][0] - t_mean) < 1e-4, "Global precision wrong"
        assert (c_result_df["precision_at_k"][0] - t_1) < 1e-4, "Count == 1 precision wrong"
        assert (c_result_df["precision_at_k"][1] - t_2) < 1e-4, "Count == 2 precision wrong"
        assert (c_result_df["precision_at_k"][2] - t_3) < 1e-4, "Count == 3 precision wrong"



def test_recall_at_k():

    m_p = RecallAtK(k=3, min_score=0.0)
    with pytest.raises(Exception):
        m_p(raw_predictions=multi_raw_preds, usecase=MockMultiUseCase(), documents=multi_documents[1:])

    for k, score in [(10, 0.5), (10, 0.0),
                     (3, 0.5), (3, 0.0),
                     (2, 0.5), (2, 0.0), (2, 0.9),
                     (1, 0.5), (1, 0.0), (1, 0.9)]:

        m_p = RecallAtK(k=k, min_score=score)
        assert m_p.name == f"recall@k (k={k}, min-score={score:.2f})", "Wrong metric name"

        c_result = m_p(raw_predictions = multi_raw_preds, usecase = MockMultiUseCase(), documents = multi_documents)
        r_result = get_multi_matches(k, multi_documents, multi_raw_preds, threshold=score)

        total = lambda r: max(1, min(len(r[0]), k))

        t_mean = np.mean([len(r[1])/total(r) for r in r_result])
        t_1 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 1])
        t_2 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 2])
        t_3 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 3])

        assert (c_result["mean"] - t_mean) < 1e-4, "Global recall wrong"
        assert (c_result["mean_per_true_count"][1] - t_1) < 1e-4, "Count == 1 recall wrong"
        assert (c_result["mean_per_true_count"][2] - t_2) < 1e-4, "Count == 2 recall wrong"
        assert (c_result["mean_per_true_count"][3] - t_3) < 1e-4, "Count == 3 recall wrong"


def test_recall_at_k_df():

    m_p = RecallAtK(k=3, min_score=0.0)
    with pytest.raises(Exception):
        m_p(raw_predictions=multi_raw_preds, usecase=MockMultiUseCase(), documents=multi_documents[1:])

    for k, score in [(10, 0.5), (10, 0.0),
                     (3, 0.5), (3, 0.0),
                     (2, 0.5), (2, 0.0), (2, 0.9),
                     (1, 0.5), (1, 0.0), (1, 0.9)]:

        m_p = RecallAtK(k=k, min_score=score)
        assert m_p.name == f"recall@k (k={k}, min-score={score:.2f})", "Wrong metric name"

        c_result_df = m_p(raw_predictions = multi_raw_preds, usecase = MockMultiUseCase(), documents = multi_documents, return_dataframe=True)
        r_result = get_multi_matches(k, multi_documents, multi_raw_preds, threshold=score)

        total = lambda r: max(1, min(len(r[0]), k))

        t_mean = np.mean([len(r[1])/total(r) for r in r_result])
        t_1 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 1])
        t_2 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 2])
        t_3 = np.mean([len(r[1])/total(r) for r in r_result if len(r[0]) == 3])

        assert (c_result_df["overall"][0] - t_mean) < 1e-4, "Global recall wrong"
        assert (c_result_df["recall_at_k"][0] - t_1) < 1e-4, "Count == 1 recall wrong"
        assert (c_result_df["recall_at_k"][1] - t_2) < 1e-4, "Count == 2 recall wrong"
        assert (c_result_df["recall_at_k"][2] - t_3) < 1e-4, "Count == 3 recall wrong"


def test_score_histogram():

    for bins in range(1, 20):
        m_h = ScoreHistogram(bins=bins)
        assert m_h.name == f"score-histogram ({bins} bins)", "Wrong metric name"

        result = m_h(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents)
        parts  = [float(r.split()[-1].strip("%"))/100 for r in result.split("\n")]
        limits = [r.replace(" - ","_").split()[0].strip("]:(").split("_") for r in result.split("\n")]

        assert len(result.split("\n")) == bins, "Wrong number of buckets"
        assert abs(sum(parts) - 1) < 1e-4, "Does not sum up to 1"
        assert all([limits[i][0] == limits[i-1][1]for i in range(1, len(limits))]), "Limits do not match up"


def test_score_histogram_df():

    for bins in range(1, 20):
        m_h = ScoreHistogram(bins=bins)
        assert m_h.name == f"score-histogram ({bins} bins)", "Wrong metric name"

        result_df = m_h(raw_predictions=raw_preds, usecase=mock_uc_1, documents=documents, return_dataframe=True)
        parts  = result_df.relative_frequency.to_list()
        limits = [r.replace(" - ","_").split()[0].strip("]:(").split("_") for r in result_df.bin.to_list()]

        assert result_df.shape[0] == bins, "Wrong number of buckets"
        assert abs(sum(parts) - 1) < 1e-4, "Does not sum up to 1"
        assert all([limits[i][0] == limits[i-1][1]for i in range(1, len(limits))]), "Limits do not match up"

def test_roc_auc():

    m_r = RocAuc()
    assert m_r.name == "roc-auc", "Wrong metric name"

    np.random.seed(42)
    for _ in range(50):

        preds = []
        for j in range(len(documents)):
            score = np.random.random()
            preds.append([RawPrediction(label="a", label_id=int(score >= 0.5), score=score)])

        result = m_r(raw_predictions=preds, usecase=mock_uc_1, documents=documents)
        c_rauc = float(result.split()[-1].strip("%"))/100
        r_rauc = get_roc_auc(documents, preds)

        assert abs(c_rauc - r_rauc) < 1e-3, "ROC-AUC is wrong"


def test_roc_auc_df():

    m_r = RocAuc()
    assert m_r.name == "roc-auc", "Wrong metric name"

    np.random.seed(42)
    for _ in range(50):

        preds = []
        for j in range(len(documents)):
            score = np.random.random()
            preds.append([RawPrediction(label="a", label_id=int(score >= 0.5), score=score)])

        result_df = m_r(raw_predictions=preds, usecase=mock_uc_1, documents=documents, return_dataframe=True)
        c_rauc = result_df.loc[0,"ROC-AUC"]
        r_rauc = get_roc_auc(documents, preds)

        assert abs(c_rauc - r_rauc) < 1e-3, "ROC-AUC is wrong"
