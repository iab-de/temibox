# Schritt für Schritt zum Text Mining mit der Temi-Box

<!--
Wir geben hier einen Überblick über
- [Schritt 1: Zielsetzung und Aufgabenstellung klären](03_step_by_step.md#schritt-1:-zielsetzung-und-aufgabenstellung-klären)
- [Schritt 2: Fachdomäne definieren](03_step_by_step.md#schritt-fachdomäne-definieren)
- [Schritt 3: Komponenten festlegen](03_step_by_step.md#Temi-Box-Pipeline)
- [Schritt 4: Pipeline konfigurieren](03_step_by_step.md#Ordnerstruktur)
- [Schritt 5: Pipeline trainieren und exportieren](03_step_by_step.md#Funktionsweise)
-->
### Schritt 1: Zielsetzung und Aufgabenstellung klären

**Zielsetzung**

Zu Beginn des Projekts klären Sie zunächst grundlegende Fragen, wie das Ziel des Projekts, nutzbare Datenquellen oder Möglichkeiten zur Evaluation der Ergebnisse.

**Aufgabenstellung**

Danach legen Sie fest, ob sich Ihre Fragestellung als Textklassifizierungsaufgabe festlegen lässt: 

- Was sind die `Label`? Bei der Textklassifizierung werden Texten vordefinierte Kategorien oder Labels zugewiesen. Es muss also klar sein, welche Zielvariable vorhergesagt werden soll und wie sie extrahiert werden kann, damit sie als Label verwendet werden kann. Die Temi-Box ist auch für Fragestellungen ohne Labels, wie beispielsweise Clustering, grundsätzlich geeignet. Allerdings geht die Dokumentation nicht im Detail darauf ein.

- Was ist das `Document`? Unter Document versteht die Temi-Box alle Text-Informationen zu einer Beobachtung, die für die Prognose relevant sind, mit Ausnahme der Labels.

- Welche Art der Textklassifizierung soll eingesetzt werden? Die Temi-Box unterscheidet vier Arten der Textklassifizierung. Welche für Sie relevant ist, können Sie der Grafik entnehmen:

![Arten der Textklassifikation](assets/textklassifikation_arten.svg)


**Empfehlungen zu den Daten**
- Die Trainingsdaten sollten so vielfältig und komplex sein wie die echten Daten. Das gilt beispielsweise in Bezug auf Länge, Stil und Formulierungen.
- Die Beispiele sollten für einen Menschen einfach zu kategorisieren sein. Mehrdeutige Beispiele sind meistens auch für automatisierte Modelle schwer zu interpretieren.
- Je mehr Beispiele pro Label im Training verwendet werden, desto genauer ist in der Regel die Vorhersage.
- Die Häufigkeit, mit der ein Label in den Trainingsdaten verwendet wird, sollte von Label zu Label nicht zu stark variieren. Um das Modell zu verbessern, können sehr selten benutzte Labels entfernt oder unter einer allgemeinen Kategorie wie "Sonstige" zusammengefasst werden.
- Beispiele, die zu keinem spezifischen Label passen, sollten möglichst einem Label wie "Sonstige" zugeordnet werden, um zu verhindern, dass ungeeignete Beispiele anderen Labels zugewiesen werden.

**Die nächsten Schritte**

Wenn Sie Texte klassifizieren möchten und die Empfehlungen zu den Daten überwiegend einhalten, können Sie für einen schnellen Einstieg oder Proof-of-Concepts den [Blueprint TextClassification](04_blueprints.md) nutzen, also eine Vorlage für eine einfache Textklassifikation.
Für die Nutzung des Blueprints liefern Sie lediglich die eigenen Daten zu und bereiten sie auf. 

Wenn Sie selbst entscheiden möchten, welche Komponenten Sie wie einsetzen, folgen Sie den weiteren Schritten.

### Schritt 2: Fachdomäne definieren

Wenn für Ihren Anwendungsfall der Blueprint nicht ausreicht, können Sie eine Pipeline mit Standardkomponenten erstellen.
Sie starten mit der Beschreibung der Fachdomäne. Wie Sie dabei vorgehen, ist unter [Domain](05_domain.md) beschrieben.


### Schritt 3: Komponenten festlegen

Im nächsten Schritt legen Sie fest, welche Komponenten Sie in Ihrer Pipeline verwenden möchten. Dafür stehen zahlreiche Standardkomponenten zur Verfügung. 
Falls Ihnen die Standardkomponenten nicht ausreichen, können Sie auch individuelle Komponenten definieren. Infos dazu finden Sie unter [Erweiterbarkeit](08_extensibility.md).

- Pipeline und Fachdomäne werden für alle Machine Learning Pipelines benötigt.

```python
# Standard-Pipeline inklusive Performance Evaluator:
from temibox.pipeline import StandardPipeline
from temibox.evaluation import Evaluator

# Klassen zur Darstellung der Fachdomäne
from temibox.domain import Document, Label, UseCase, DataLoader
```


- Für traditionelles Text Mining wird der TF-IDF-Embedder verwendet, für Deep Learning Methoden der BERT-Embedder.
BERT braucht zusätzlich den Standard-Trainer. Darüber hinaus muss ein vortrainiertes Modell heruntergeladen werden, z. B. RoBERTa oder DistilBERT.

```python
# traditionelles Text Mining mit TF-IDF:
from temibox.embedder import tfidf

# Deep Learning mit BERT-Embedder und Standard-Trainer für neuronale Netze bzw. BERT-Modelle:
from temibox.embedder import BertEmbedder
from temibox.trainer import StandardTrainer
```


- Alle weiteren Komponenten sind von der Aufgabenstellung abhängig. Dazu zählen die Art des Klassifizierers, die Verlustfunktion (nur relevant für BERT-Embedder) und die Performance-Metriken.
Klassifizierer setzen teilweise bestimmte Embedder voraus. Deshalb ist in der folgenden Tabelle der Klassifizierer zusammen mit dem Embedder angegeben.

| Aufgabe                                | Embedder + Klassifizierer                         | Verlustfunktion<br>(nur relevant für<br>BERT-Embedder) | Performance-Metrik<br>(nur für Klassifikation)                                         |
|----------------------------------------|---------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------|
| binäre Klassifikation                  | TF-IDF + KNN<br>BERT + BinaryClassifier           | BinaryLoss                                             | Accuracy<br>ConfusionMatrix (inkl. Precision, Recall, F1)<br>ROC-AUC<br>ScoreHistogram |
| multi-class Klassifikation             | TF-IDF + KNN<br>BERT + MultinomialClassifier      | MultinomialLoss                                        | ConfusionMatrix (inkl. Precision, Recall, F1)<br>ROC-AUC                               |
| multi-label Klassifikation <br>binär   | BERT + BinaryClassifier<br>(nur für BERT möglich) | MultilabelBinaryLoss                                   | F1 (inkl. Precision, Recall)<br>PrecisionAtK, RecallAtK                                |
| multi-label Klassifikation multinomial | TF-IDF + KNN<br>BERT + MultinomialClassifier      | MultinomialLoss                                        | PrecisionAtK, RecallAtK                                                                |
| Clustering                             | TF-IDF + KMeans<br>BERT + KMeans                  | -                                                      | -                                                                                      |


```python
# Classifier für Text Klassifikation und Text Clustering
from temibox.model.classifier import BinaryClassifier, MultinomialClassifier, KnnClassifier
from temibox.model.classifier import KMeansCluster

# Verlustfunktionen (nur relevant für BERT-Embedder):
from temibox.losses import BinaryLoss, MultinomialLoss, MultilabelBinaryLoss, TripletLoss

# Performance Metriken (abhängig von der Aufgabe):
from temibox.evaluation.metric import Accuracy, ConfusionMatrix, F1, PrecisionAtK, RecallAtK, RocAuc, ScoreHistogram
```


### Schritt 4: Pipeline konfigurieren

Im zweiten Schritt setzen Sie die zuvor festgelegten Komponenten in die Pipeline `temibox.pipeline.StandardPipeline` ein. 
Die wichtigsten Methoden dazu sind:

- **add_usecase()**

> ```python
> StandardPipeline.add_usecase(usecase)
> ```
>
> Fügt der Pipeline einen Anwendungsfall hinzu. 
> Achtung: Jede Pipeline muss mindestens einen Anwendungsfall enthalten (Subtyp von `temibox.domain.UseCase`).
>
>> Parameter
>> - usecase (UseCase): Name des neuen Anwendungsfalls. Die Namen verschiedener Anwendungsfälle müssen eindeutig sein.


- **add_step()**

> ```python
> StandardPipeline.add_step(name, step, usecases=None, dependencies=None)
> ```
> Fügt einen zusätzlichen Schritt zur Pipeline hinzu. 
> Achtung: Jede Pipeline muss mindestens einen Schritt enthalten. 
> 
> Parameter
> - name (str): frei wählbarer Name der Komponente, die hinzugefügt werden soll. Die Namen müssen eindeutig sein.
> - step (PipelineStep): Komponente, die hinzugefügt werden soll
> - usecases (Liste der Usecases, optional): Liste der Instanzen der Usecases, für die die Komponente genutzt werden soll. Standardmäßig wird die Komponente für alle Usecases genutzt. Die Komponente muss mindestens einem Usecase zugeordnet sein.
> - dependencies (list, optional): Liste der Steps, von denen der neue Step abhängig ist.  Es dürfen nur Abhängigkeiten zu bereits aufgenommenen Schritten festgelegt werden (keine zyklischen Abhängigkeiten).
Falls Abhängigkeiten angegeben wurden, dann muss die Liste *alle* Abhängigkeiten des Schrittes enthalten. Standardmäßig wird der Schritt von allen bisher aufgenommenen Schritten abhängig gemacht.
>
> Hinweis: Es dürfen mehrere Instanzen der gleichen Klasse aufgenommen werden (z.B. mehrere Embedder, Modelle usw. gleichen Typs). 
> In diesem Fall müssen Abhängigkeiten (dependencies) angegeben werden, ansonsten überschreibt die Ausgabe der letzten Komponente der Art die Ausgaben der anderen "Geschwister"-Komponenten. Alternativ können die Komponenten zu verschiedenen Anwendungsfällen zugeordnet werden.
> 
> Ob und wie Komponenten von einander abhängen sollen, ist von der Fragestellung abhängig. 
> Manchmal ist es sinnvoll, dass z.B. verschiedene Klassifizierer den gleichen Embedder nutzen, da sich dadurch die Prognosequalität der gesamten Pipeline verbessert. 
> Falls aber die Anwendungsfälle zu unterschiedlich sind, könnte das gleiche Vorgehen auch schädlich und mehrere unabhängige Pipelines sinnvoller sein.
>
> Beispiel
> ```python
> StandardPipeline.add_step("modell", MultinomialClassifier(multilabel=True), usecases=[uc_1], dependencies=["bert-embedder-1"])
> ``` 


Hier ist ein Beispiel für eine Umsetzung einer einfachen Pipeline:

```python
from temibox.pipeline import StandardPipeline
from temibox.embedder import BertEmbedder
from temibox.model.classifier import MultinomialClassifier
from temibox.trainer import StandardTrainer

# Fachwissen
from .domain import PublikationenLader, ThemenKlassifizierung

# Pipelinedefinition
pipeline = (StandardPipeline()
                # Fachwissen: Anwendungsfall
                .add_usecase(ThemenKlassifizierung())
    
                # Fachwissen: Daten
                .add_step("daten", PublikationenLader())
    
                # Embedder (basiert auf DistilBERT)
                .add_step("embedder", BertEmbedder("modelle/distilbert"))                
    
                # Klassifizierer                           
                .add_step("modell", MultinomialClassifier(multilabel=True))
    
                # Trainer
                .add_step("trainer", StandardTrainer()))
```

Hier ist ein Beispiel für eine Umsetzung einer Pipeline mit mehreren Anwendungsfällen und Komponenten gleicher Art
(siehe auch [06_advanced_pipelines](06_advanced_pipelines.md)):

```python
from temibox.pipeline import StandardPipeline
from temibox.embedder import BertEmbedder
from temibox.model.classifier import BinaryClassifier, MultinomialClassifier
from temibox.trainer import StandardTrainer

# Fachwissen
from .domain import DatenLader, MeinAnwendungsfall_1, MeinAnwendungsfall_2

# Pipelinedefinition
pipeline = (StandardPipeline()
                # Fachwissen: zwei Anwendungsfälle
                .add_usecase(uc_1 := MeinAnwendungsfall_1())
                .add_usecase(uc_2 := MeinAnwendungsfall_2())
    
                # Fachwissen: Daten
                .add_step("daten", DatenLader())
    
                # Zwei BERT-Embedder
                .add_step("bert-embedder-1", BertEmbedder("modelle/distilbert"))
                .add_step("bert-embedder-2", BertEmbedder("modelle/distilbert"), usecases=[uc_2])
    
                # Drei Klassifizierer (zwei nutzen den ersten Embedder, das dritte den zweiten)           
                .add_step("mein-modell-1", BinaryClassifier(multilabel=True), dependencies=["daten", "bert-embedder-1"])
                .add_step("mein-modell-2", MultinomialClassifier(multilabel=True), dependencies=["daten", "bert-embedder-1"])
                .add_step("mein-modell-3", MultinomialClassifier(multilabel=True), dependencies=["daten", "bert-embedder-2"], usecases=[uc_2])
    
                # Trainer für alle Embedder und Klassifizierer
                .add_step("trainer", StandardTrainer()))
```


### Schritt 5: Pipeline trainieren und exportieren

Für das Training der Pipeline verwenden Sie die `.train()`-Methode. Es ist empfehlenswert, die Komplexität der Pipelines schrittweise zu erhöhen. Zunächst sollten Sie sicherstellen, dass die Pipeline korrekt funktioniert und Modelle erfolgreich trainiert werden (der Wert der Verlustfunktion sinkt und die Performancemetriken steigen). Anschließend können Sie Fine-Tuning-Techniken einsetzen, um die Modellleistung weiter zu optimieren.

```python
# Training Beispiel 1
pipeline.train()

# Training Beispiel 2
pipeline.train(pub_json_dir = "data/json/")
```

Beispiel für Vorhersage und Export der Pipeline:

```python
# Vorhersage
with pipeline.modes(inference = True, cuda = True):
  vorhersage = pipeline.predict(title    = "Beispieltitel",
                                abstract = "Kurze Zusammenfassung",
                                keywords = ["Schlagwort 1", "Schlagwort 2"])

# Export
pipeline.export(folder = r"C:\poc\getting_started")
```



