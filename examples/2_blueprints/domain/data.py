import ast
import pandas as pd
from tqdm import tqdm

from temibox.domain import LabelDescription
from temibox.blueprint import TextClassification

# Labels CSV file containing columns:type
# - label_id: int
# - label:    str
# - weight:   float
#
# Example rows:
#
#label_id;label;weight
#1322;"Target label A";93
#1636;"Target label B";12
#1771;"Target label C";551
def load_labels(path_csv: str) -> list[LabelDescription]:
    df = pd.read_csv(path_csv, sep=";", decimal=",", encoding="utf-8")

    labels = []
    for _, row in tqdm(df.iterrows(), "Processing dataframe"):
        labels.append(LabelDescription(label_id = row.label_id,
                                       label    = row.label,
                                       weight   = row.weight))

    return labels

# Data CSV file containing columns: type
# - text:    str
# - labels:  list[int]
#
# Example rows:
#
#text;labels
#"This is a text";[3,4,18]
#"This is another text";[6]
def load_data(path_csv: str, labels: list[LabelDescription], max_labels: int = 10) -> list[TextClassification.Document]:
    df = pd.read_csv(path_csv, sep=";", decimal=",", encoding="utf-8")
    label_ids = {ld.label_id for ld in labels}

    docs = []
    for _,row in tqdm(df.iterrows(), "Processing dataframe"):
        i_label_ids = [x for x in ast.literal_eval(row.labels) if x in label_ids]
        if not len(i_label_ids) or len(i_label_ids) > max_labels:
            continue

        docs.append(TextClassification.Document(text=row.text,
                                                label_ids=i_label_ids))

    return docs
