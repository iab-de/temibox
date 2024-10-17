import os
import sys
import pytest
import tempfile
sys.path.append(f"{os.getcwd()}/examples/2_blueprints")

from temibox.blueprint import TextClassification
from domain import load_data, load_labels

# Path to pretrained BERT-based model
PRETRAINED_DIR = os.getenv("PRETRAINED_DIR")

# Path to train/test/labels data
DATA_DIR = os.getenv("DATA_DIR")

# Data
labels          = load_labels(f"{DATA_DIR}/labels.csv")
train_documents = load_data(f"{DATA_DIR}/train_data.csv", labels)
test_documents  = load_data(f"{DATA_DIR}/test_data.csv", labels)


def test_usecase():
    task = TextClassification.TASK.MULTILABEL_MULTINOMIAL

    # Initialize pipeline
    cls = TextClassification(pretrained_bert_dir=PRETRAINED_DIR,
                             classification_task=task,
                             create_checkpoints=True,
                             train_vali_split=0.9,
                             labels=labels.copy(),
                             layer_dimensions=[2048])

    usecase = cls.usecase

    assert usecase.name == "text-classification", "wrong usecase name"
    assert len(usecase.get_usecase_label_dict()) > 0, "label dict empty"
    assert len(usecase.get_usecase_label_dict())  == len(usecase.get_usecase_labels()), "Invalid label list"

    doc = train_documents[-1]
    doc_labels = usecase.get_document_labels(doc)

    assert len(usecase.get_document_body(doc)) > 0, "Body extraction fails"
    assert all([x in usecase.get_usecase_labels() for x in doc_labels]), "Unknown labels"

def test_train_and_predict():

    for task in [TextClassification.TASK.MULTILABEL_MULTINOMIAL,
                 TextClassification.TASK.MULTILABEL_BINARY]:

        # Initialize pipeline
        cls = TextClassification(pretrained_bert_dir=PRETRAINED_DIR,
                                 classification_task=task,
                                 create_checkpoints=True,
                                 train_vali_split=0.5,
                                 labels=labels.copy(),
                                 layer_dimensions=[8])

        usecase = cls.usecase
        ulabels = usecase.get_usecase_labels()

        cls.train(documents = train_documents[:32],
                  max_epochs = 1)

        # Predict
        max_predictions = 4
        preds = cls.predict(document=train_documents[:8], max_predictions = max_predictions)

        assert len(preds) == 8, "Invalid prediction count"

        for pred in preds:
            assert len(pred.payload_raw) == len(ulabels), "Invalid number of predictions"
            assert pred.payload.shape[0] <= max_predictions, "Invalid number of rows in payload"


        # Evaluate
        result = cls.evaluate(documents = test_documents[:8], return_dataframe=True)
        assert len(result) == len(cls.pipeline.get_step("evaluator")._metrics), "Invalid number of evaluation results"

        # Export

        export_path = f"{tempfile.gettempdir()}/test_csl.pkl"
        cls.export(full_path = export_path)

        del cls

        cls_loaded = TextClassification.load(export_path)

        preds_loaded = cls_loaded.predict(document=train_documents[:8], max_predictions=max_predictions)

        assert len(preds_loaded) == len(preds), "Invalid prediction count (loaded)"

        for i in range(len(preds)):
            assert preds_loaded[i].payload.equals(preds[i].payload), f"Payload of {i+1} prediction wrong"

        result_loaded = cls_loaded.evaluate(documents=test_documents[:8], return_dataframe=True)
        for i in range(len(result)):
            assert result[i].equals(result_loaded[i]), f"Evaluation results for {i+1} deviate"


def test_different_tasks():

    for task in TextClassification.TASK:

        TextClassification(pretrained_bert_dir=PRETRAINED_DIR,
                           classification_task=task,
                           create_checkpoints=True,
                           train_vali_split=0.5,
                           labels=labels.copy(),
                           layer_dimensions=[8])

    with pytest.raises(Exception):
        TextClassification(pretrained_bert_dir=PRETRAINED_DIR,
                           classification_task="does not work",
                           create_checkpoints=True,
                           train_vali_split=0.5,
                           labels=labels.copy(),
                           layer_dimensions=[8])