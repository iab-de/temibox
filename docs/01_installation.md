# Installation

Die Installation der Temi-Box besteht aus drei Schritten:
- [Einrichten einer virtuellen Umgebung](01_installation.md#einrichten-einer-virtuellen-umgebung)
- [Installation der Temi-Box](01_installation.md#installation-der-temi-box)
- [Download vortrainierter Modelle und Pipelines](01_installation.md#download-vortrainierter-modelle-und-pipelines)

Hinweis: Die Temi-Box wurde für Windows als Betriebssystem entwickelt. Die folgende Beschreibung dokumentiert deshalb nur das Vorgehen in Windows-Umgebungen.


## Einrichten einer virtuellen Umgebung

Richten Sie zunächst eine virtuelle Umgebung ein. 

Dieser Schritt ist optional, aber dringend empfohlen, um 
- Konflikte mit anderen Projekten oder Bibliotheken zu vermeiden,
- durch die Dokumentation der verwendeten (externen) Bibliotheken die Reproduzierbarkeit des Projekts auf anderen Systemen zu gewährleisten (z. B. für den Austausch von Codes in gemeinsamen Forschungsprojekten) und
- die Entwicklungsumgebung frei von unnötigen Paketen zu halten.

Eine virtuelle Umgebung isoliert Ihren Code von allen anderen, auch global installierten Paketen, so dass Ihr Projekt nur auf die Bibliotheken
zugreifen kann, die Sie in der virtuellen Umgebung manuell installiert haben.
Mit dem python Paketmanager (`pip freeze`) können Sie einfach die Liste der verwendeten Bibliotheken abrufen 
und bei Bedarf zusammen mit dem Source Code verteilen (standardmäßig in der Datei `requirements.txt`).

Wie eine virtuelle Umgebung erzeugt werden kann, hängt von Ihrem Betriebssystem
und ggf. der IT-Sicherheitsrichtlinien ab. Auf Windows-Systemen wird oft die 
Open-Source-Distribution `Anaconda` eingesetzt:

```shell
conda create --offline -p <Pfad> // Erzeugung der virtuellen Umgebung
conda activate <Pfad>            // Aktivierung der virtuellen Umgebung 
conda install python=3.10        // Installation von python (in diesem Beispiel Version 3.10)
pip install -r requirements.txt  // Installation weiterer Pakete 
```
Abhängig von bestehenden IT-Sicherheitsrichtlinien kann es notwendig sein, bestimmte Proxy Einstellungen oder conda/pip Repositories anzugeben.


## Installation der Temi-Box

Laden Sie den `temibox` Source-Code entweder mittels `git clone` oder per `http` herunter und übergeben Sie `pip` den Pfad
zum Ordner:

```shell 
pip install <lokaler Pfad>
```

## Download vortrainierter Modelle und Pipelines

Bei den meisten Temi-Box-Pipelines können Sie vortrainierte, BERT-kompatible Modelle einsetzen. 
Sie finden viele aktuelle Modelle bei [huggingface](https://huggingface.co), 
z.B. [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased/tree/main). 
Laden Sie diese Modelle manuell herunter. 

Für bestimmte Aufgaben (besonders im Bereich Preprocessing) können auch vortrainierte Pipelines von [spaCy](https://spacy.io/) eingesetzt werden, z.B. [de_core_news_lg](https://huggingface.co/spacy/de_core_news_lg/tree/main).
Auch hier können Sie die entsprechenden Artefakte manuell herunterladen.

