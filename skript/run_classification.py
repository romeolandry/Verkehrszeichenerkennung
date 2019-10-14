import os
import sys
import tensorflow as tf
from daten.Data_vorbebreitung import Data_vorbebreitung
from models.Classification import Classification, Classification_test
from tensorflow.python.keras import (losses,
                                     optimizers,
                                     metrics,
                                     regularizers)

base_path = os.getcwd()
path_to_gtsrb = os.path.join(base_path, '../Daten/Final_Training/Images')
path_to_class_beschreibung = os.path.join(base_path, '../utils/' +
                                          'Text_Beschreibung.csv')
path_to_model = os.path.join(base_path, '../Model_weight/')

IMG_SIZE = 48
NUM_BATCH = 64
NUM_EPOCHS = 1000
verborse = 1
validation_split = 0.2
lernrate = 0.001

optimizer = optimizers.Adamax(lernrate)
# categorical_ crossentropie ist für die
# Klassifikationsaufgabe geeignet
loss = 'categorical_crossentropy'
metrics = ["accuracy"]


# image und Labels zum Trainen werden gelesen
data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, IMG_SIZE,
                                       path_to_class_beschreibung)
print("Daten werden eingelesen.....")
train_images, train_labels, classes = data_vorbereitung.load_roadsigns_data()
print("Daten eingelesen!")
# data_vorbereitung.display_roadsign_classes(train_images, 10)
# print("beispiel von Labelskalssen: {}".format(train_labels[3]))

# Vorbereitung des Models
print("Aufbau des Models!")
train_model = Classification(path_to_model, IMG_SIZE, loss, optimizer, metrics,
                             verborse, validation_split, classes,
                             NUM_BATCH, NUM_EPOCHS)
model = train_model.build_model()
print("Training")
train_model.train_model(model, train_images, train_labels)

# Test model
test_model = Classification_test(path_to_model, IMG_SIZE)

