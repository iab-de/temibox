import os
import sys
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

# TextClassification blueprint supports following classification tasks:
# - TextClassification.TASK.BINARY                 ("Text" -> one of two labels)
# - TextClassification.TASK.MULTICLASS_MULTINOMIAL ("Text" -> one of many labels (fixed set))
# - TextClassification.TASK.MULTILABEL_BINARY      ("Text" -> some of many labels (variable set))
# - TextClassification.TASK.MULTILABEL_MULTINOMIAL ("Text" -> some of many labels (fixed set))
#
# This example uses multiclass/multilabel data, thus only the last
# two tasks are applicable
task = TextClassification.TASK.MULTILABEL_MULTINOMIAL

# Initialize pipeline
cls = TextClassification(pretrained_bert_dir = PRETRAINED_DIR,
                         classification_task = task,
                         create_checkpoints  = True,
                         train_vali_split    = 0.9,
                         labels              = labels.copy(),
                         layer_dimensions    = [2048])

# Train pipeline
cls.train(documents = train_documents)

# Evaluate performance metrics
metrics = cls.evaluate(documents = test_documents)
metrics = cls.evaluate(documents = test_documents, return_dataframe = True)

# Create prediction
preds = cls.predict(document = train_documents[0])
preds[0].payload

