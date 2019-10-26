# Verkehrszeichnerkennung

## Beschreibung

Die Daten, die in diesem projekt benuzt werden, wurde von INI(Institu für Informatik) RUB
veröffentlichen.

- Jede Bild beinhaltet nur eine klasse
- Es gibt ungewähr 40 Klassen

## Requirement

- numpy==1.17.2
- pandas==0.25.1
- Pillow==6.2.0
- scikit-learn==0.16.1
- tensorflow-gpu==2.0.0a0
- matplotlib

## training des Models

Das Model lässig durch run_classification.py
Training auf GTSRB-Daten von INI.

- python run_classification.py -p [Pfad zur Trainingsdaten] -d [Pfad zur beschreibung-Datei] -s [pfad zur Speicherung des Models]

- Pfad zur Trainingsdaten: muss gegeben werden.
- Pfad zur beschreibung-Datei : Es gibt eine Csv-Beschreibung im Daten/utils, die standardmäßig ausgewält werden.
- pfad zur Speicherung des Models: Es wird standardmäßig im Models/keras-Model gespeichert.

## Test Keras-Model

## Umwandlung von Keras-Model zu TensortRT-Model

Das trainierte Keras-Model soll voraussetztliche mit unserem Model aufgebaut werden sonst wird ein fehler aufgetreten.
Fall ein andere keras - Model benutzt sein soll, muss dieses den glieche Input- und Outputname haben bzw. "conv2d_input:0" und "dense_1/Softmax:0" haben.
mit dem Befehl:

- python run_convert_keras_trt.py -p [Pfad zum keras-Model]
- Pfad zum keras-Model: soll .h5-Datei sein und gegeben werden

## Issue
