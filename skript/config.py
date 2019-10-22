import os
import datetime
"""
Die datei soll alle mögliche Globale Variable beinaltet
"""
# Daten

base_path = os.getcwd()

path_to_class_beschreibung = os.path.join(base_path, '../Daten/utils/' +
                                          'Text_Beschreibung.csv')
classes_count = 43

keras_model_name = "keras_model"
# Pfad zur Speicherung des Keras-Models
path_to_model = os.path.join(base_path, '../Models/Keras-Model/' +
                             keras_model_name + ' _{}.h5'.format(
                                 datetime.datetime.now().strftime(
                                     "%Y_%m_%d_%H_%M_%S")))

# Model Training und Test
IMG_SIZE = 48
NUM_BATCH = 64
NUM_EPOCHS = 1000
verborse = 1
validation_split = 0.2
lernrate = 0.001
patience = 200


loss = 'categorical_crossentropy'
metrics = ["accuracy"]

# Tensorflow -> tensortRT
"""
from pfad des keras -model wird die Name extrahieren
bsp. ".../model_test.h5 wird "model_test" extrahieren.
tensorflow-model wird dann gennant tf_model_from_"model_test"
    ordner from Tensor-Model wird so aussehen:
        tf_model_from_model_test
                |_variable
                |_saved_model.pb
                |_tf_frozen_from_"model_test.pb"
    und RT-Model wird so aussehen
        RT-Model
            |_trt_frozen_from_"model.pb"
"""
keras_model_name = ""

tf_model_from_keras = "tf_model_from_" + keras_model_name
tf_frozen_model_name = "tf_frozen_from_" + keras_model_name + ".pb"
trt_frozen_name = "rt_frozen_from_" + keras_model_name + ".pb"

path_tf_model = os.path.join(base_path, '../Models/Tensor-Model/' +
                             tf_model_from_keras)
path_tf_frozen_model = os.path.join(path_tf_model, tf_frozen_model_name)
# path_h5_model = os.path.join(base_path, '../Models/Keras-Model/')
path_trt_frozen_model = os.path.join(base_path, '../Models/RT-Model/' +
                                     trt_frozen_name)

# config gpu
# tensorflow allocation memory
tf_aloc = 0.5

max_batch_size = 2
max_workspace_size_bytes = 2*(10**9)
# FP32 FP16 int8
precision_mode = "FP32"

# tensorflow model
output_model = {}
input_model = {}
