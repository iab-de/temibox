# Vorlage / Blueprint zur Textklassifikation

Für den schnellen Einstieg in das Textmining oder Proof-of-Concepts bietet die Temi-Box Blueprints, also Vorlagen für bestimmte Anwendungsfälle. 
Für die Nutzung der Blueprints müssen lediglich die eigenen Daten zugeliefert und aufbereitet werden.
Mit der Vorlage `temibox.blueprint.TextClassification` lassen sich Textklassifizierungsaufgaben mit wenig Aufwand umsetzen.


### Schritt 1: Analyse der Voraussetzungen 

Um entscheiden zu können, ob Sie den Blueprint einsetzen können, beantworten Sie diese Fragen:
- **Was sind die Labels?** 
Bei der Textklassifizierung werden Texten vordefinierte Kategorien oder Labels zugewiesen. Es muss also klar sein, was die Zielvariable ist und wie sie sich extrahieren lässt, damit sie als Label verwendet werden kann.
- **Was ist das Document?** Unter Document versteht die Temi-Box alle Text-Informationen zu einer Beobachtung, die für die Prognose relevant sind, mit Ausnahme der Labels.
- **Welche Art der Textklassifizierung soll eingesetzt werden?** Die Temi-Box unterscheidet die vier Arten der Textklassifizierung. Welche für Sie relevant ist, können Sie der Grafik entnehmen:

![Arten der Textklassifikation](assets/textklassifikation_arten.svg)
- **Wie lassen sich die Daten importieren?** Diese Dokumentation liefert Umsetzungbeispiele für Rohdaten im csv- und JSON-Format. Die Rohdaten können aber auch aus anderen Quellen eingelesen werden.


### Schritt 2: Vorbereitung der Daten

Zur Vorbereitung der Daten erstellen Sie eine Liste der Labels und Dokumente. Dazu definieren Sie eine neue Datenklasse, z. B. `data` mit diesen Funktionen:
- **`get_labels`**: erstellt eine Liste der Labels für den Blueprint (`list[LabelDescription]`)
- **`get_documents`** erstelle eine Liste der Dokumente für den Blueprint (`list[TextClassification.Document]`)


Beispiel, bei dem die benötigten Daten im csv-Format im Ordner `data/` verfügbar sind:

```python
import ast
import pandas as pd
from tqdm import tqdm
from temibox.domain import LabelDescription
from temibox.blueprint import TextClassification

class Data:

    @staticmethod
    def load_labels(path_csv: str) -> list[LabelDescription]:
        df = pd.read_csv(path_csv="data\labels.csv", sep=";", decimal=",", encoding="utf-8")
    
        labels = []
        for _, row in tqdm(df.iterrows(), "Processing dataframe"):
            labels.append(LabelDescription(label_id = row.label_id,
                                           label    = row.label,
                                           weight   = row.weight))
        return labels

    @staticmethod
    def load_data(path_csv: str, labels: list[LabelDescription], max_labels: int = 10) -> list[TextClassification.Document]:
        df = pd.read_csv(path_csv="data\publications.csv", sep=";", decimal=",", encoding="utf-8")
        label_ids = {ld.label_id for ld in labels}
    
        docs = []
        for _,row in tqdm(df.iterrows(), "Processing dataframe"):
            i_label_ids = [x for x in ast.literal_eval(row.labels) if x in label_ids]
            if not len(i_label_ids) or len(i_label_ids) > max_labels:
                continue

            docs.append(TextClassification.Document(text=row.text,
                                                    label_ids=i_label_ids))
        return docs
```


Beispiel, bei dem die benötigten Daten im JSON-Format im Ordner `data/` verfügbar sind:

```python
import json
from temibox.blueprint import TextClassification
from temibox.domain import LabelDescription

class Data:
        
    @staticmethod
    def get_labels() -> list[LabelDescription]:
        with open("data/labels.json", "r") as f:
            labels = [LabelDescription(label_id = entry["topic_id"],
                                       label    = entry["topic"],
                                       weight   = entry["topic_abs_freq"]) for entry in json.load(f)]
        return labels            
    
    @staticmethod
    def get_documents(max_documents: int = None) -> list[TextClassification.Document]:
        documents = []
        with open("data/publications.json", "r") as f:
            for entry in json.load(f):
                doc = TextClassification.Document(text = ". ".join([entry["title"], *entry["keywords"], entry["abstract"]]),
                                                  label_ids = entry["topic_ids"])
                documents.append(doc)
                
                if max_documents is not None and len(documents) == max_documents:
                    break
        
        return documents
```

### Schritt 3: Initialisierung der TextClassification

Der Trainingsprozess zum Blueprint Textklassifikation startet mit der Initialisierung der Klasse 

**TextClassification()**

>```python
>from temibox.blueprint import TextClassification
>
>TextClassification(pretrained_bert_dir, classification_task, labels, documents = None,
>                 use_class_weights = False, train_vali_split = 0.9, layer_dimensions = None,
>                 allow_cuda = True, create_checkpoints = True)
>```
>
>
>Die wichtigsten Parameter sind:
>- **pretrained_bert_dir** (str): Verzeichnis mit einem vortrainierten Bert-Modell.
>- **classification_task**: Art der Klassifikation. Im Blueprint verfügbar sind:
>  - TextClassification.TASK.BINARY
>  - TextClassification.TASK.MULTINOMIAL
>  - TextClassification.TASK.MULTILABEL_BINARY 
>  - TextClassification.TASK.MULTILABEL_MULTINOMIAL 
>- **labels** (list): Liste der Labels. Standardmäßig definierbar als `NameDerDatenklasse.get_labels()`.
>- **documents** (list, optional): Liste der Dokumente.
>
>Weitere optionale Parameter sind:
>- **use_class_weights** (bool, optional): Angabe, ob Klassen gewichtet werden
>- **train_vali_split** (float, optional): Anteil der Daten, der für das Training (nicht zur Validierung) verwendet wird (Checkpointing und vorzeitige Unterbrechungen sind nur bei `train_vali_split < 0,99` möglich)
>- **layer_dimensions** (list, optional): Zahl der Dimensionen der Layer, die der shallow Classifier über dem BERT-Embedder verwendet. Standardwert: `32`
>- **allow_cuda** (bool, optional): Angabe, ob zum Training die GPU genutzt werden darf
>- **create_checkpoints** (bool, optional): Angabe, ob Checkpoints erstellt und ggf. beim Training wiederhergestellt werden sollen (nur wenn Validierungsdaten verfügbar sind)
>
>Beispiel:
>
>>```python
>>from temibox.blueprint import TextClassification
>>from .data import Data
>>
>>cls = TextClassification(pretrained_bert_dir = "modelle/distilbert",
>>                         classification_task = TextClassification.TASK.MULTICLASS_MULTINOMIAL,
>>                         labels    = Data.get_labels(),
>>                         documents = Data.get_documents())
>>```

### Schritt 4: Vorhersage und Export des Modells

Vorhersage und Export des Modells erfolgen durch die Anwendung der Methoden `.predict()` und `.export()`.
Die Methode `.export()` exportiert und bereinigt (optional) das Modell, um das Modell möglichst klein zu halten.

Beispiel:

```python
cls.predict(document = TextClassification.Document(text = "Klassifiziere diesen Text"))
cls.export(full_path = r"C:\poc\getting_started\textclassification.pkl")
```