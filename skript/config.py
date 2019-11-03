import os
import datetime
import random

from tensorflow.python.keras import optimizers

"""
Die datei soll alle mögliche Globale Variable beinaltet
"""
random.seed(30)
# Daten

base_path = os.getcwd()

pfad_zu_performance_model = os.path.join(base_path,
                                         '../performance/' +
                                         'test.csv')

path_to_class_beschreibung = os.path.join(base_path, '../Daten/utils/' +
                                          'Text_Beschreibung.csv')
classes_count = 43

keras_model_name = "keras_model"
# Pfad zur Speicherung des Keras-Models
path_to_model = os.path.join(base_path, '../Models/Keras-Model/')
# pfad zur Birlder
pfad_to_ergebnis_bild = os.path.join(base_path, '../Ergebnis/')
pfad_zu_logs = os.path.join(base_path, '../logs')
# Training und Test des  Models
IMG_SIZE = 48
NUM_BATCH = 64
NUM_EPOCHS = 1
verborse = 1
validation_split = 0.25
lernrate = 1e-5
patience = round(NUM_EPOCHS/NUM_BATCH)
min_delta = 0.001

# welches Model wird trainiert
# set false fall das Model nicht trainiert sein wird
simple_model = True
simple_vgg = True

# Das training wird durch gelaufen werden mit Data_generator
# set false wenn es auf Data-generator vertichtet sein werden
data_gen = True
ohne_gen = True

# Das training Parametern
loss = 'categorical_crossentropy'
metrics = ["accuracy"]
optimizer = optimizers.adam(lr=lernrate)

# config gpu
# tensorflow allocation memory
tf_aloc = 0.5

max_batch_size = 2
max_workspace_size_bytes = 2*(10**9)
# FP32 FP16 int8
precision_mode = "FP16"

# tensorflow model
output_model = {}
input_model = {}
