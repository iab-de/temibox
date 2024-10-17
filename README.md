<a><img src="docs/assets/temibox_logo.svg" align="right"></a>
# Temi-Box - Text Mining in Python

Die Temi-Box ist eine python-Bibliothek zum Erstellen und Trainieren 
von Text Mining-Pipelines mit Fokus auf Benutzerfreundlichkeit und Trennung 
von Fachwissen und NLP. 

## Modulare Komponenten
Verschiedene Methoden zum Text Mining stehen als vorgefertigte Komponenten zur Verfügung und können direkt in eine Pipeline integriert werden. Damit lassen sich Entwicklung und Implementierung zum Text Mining erheblich vereinfachen.

```python
from temibox.pipeline import StandardPipeline
from temibox.embedder import BertEmbedder
from temibox.trainer import StandardTrainer
from temibox.model.classifier import BinaryClassifier, MultinomialClassifier

# Fachwissen
from .domain import PublikationenLader, ThemenKlassifizierung

# Pipelinedefinition
pipeline = (
            StandardPipeline()
            
                # Fachwissen: Anwendungsfall
                .add_usecase(ThemenKlassifizierung())
    
                # Fachwissen: Daten
                .add_step("daten", PublikationenLader())
    
                # Embedder (basiert auf DistilBERT)
                .add_step("embedder", BertEmbedder("modelle/distilbert"))                
    
                # Klassifizierer
                .add_step("modell-1", BinaryClassifier(multilabel=True))
                .add_step("modell-2", MultinomialClassifier(multilabel=True))
    
                # Trainer
                .add_step("trainer", StandardTrainer())
)

# Training
pipeline.train(pub_path_json = "daten/json")

# Vorhersage
with pipeline.modes(inference = True, cuda = True):
  vorhersage = pipeline.predict(title    = "Beispieltitel",
                                abstract = "Kurze Zusammenfassung",
                                keywords = ["Schlagwort 1", "Schlagwort 2"])

# Export
pipeline.export(folder = r"C:\poc\getting_started")
```

Die `temibox` basiert auf bedeutenden NLP-Bibliotheken wie `pytorch`, `transformers`
und `spaCy` und bietet eine höhere und benutzerfreundlichere Abstraktionsebene.

## Dokumentation

- [Installation](docs/01_installation.md)
- [Überblick](docs/02_overview.md)
- [Erste Schritte](docs/03_step_by_step.md)
- [Vorlagen](docs/04_blueprints.md)
- [Fachdomäne](docs/05_domain.md)
- [Fortgeschrittene Pipelines](docs/06_advanced_pipelines.md)
- [Technische Details](docs/07_technical_details.md)
- [Möglichkeiten zur Erweiterung](docs/08_extensibility.md)


## Systemanforderungen

In den meisten `temibox`-Pipelines werden künstliche neuronale Netze entweder trainiert oder fine-tuned. 
Daher wird dringend empfohlen, das Training ausschließlich auf einem Rechner mit einer ausreichend großen (mindestens 8 GB VRAM) CUDA-fähigen GPU durchzuführen. Während für die Inferenz in vielen Fällen auf eine GPU verzichtet werden kann, ist sie im Trainingsprozess unerlässlich.

<a><img src="docs/assets/EU_Logo.jpg" height="125" align="left"></a>
## Acknowledgments

Die Temi-Box entstand im Projekt "Etablieren eines Standardbaukastens für Text Mining" und wurde finanziert von der Europäischen Union - NextGenerationEU. 

