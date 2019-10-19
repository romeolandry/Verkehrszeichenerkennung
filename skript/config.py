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

# Pfad zur Speicherung des Keras-Models
path_to_model = os.path.join(base_path, '../Models/Keras-Model/' +
                             'signs_model_Kalssifikation_{}.h5'.format(
                                 datetime.datetime.now().strftime(
                                     "%Y_%m_%d_%H_%M_%S")))

# Model Training und Test
IMG_SIZE = 48
NUM_BATCH = 64
NUM_EPOCHS = 1000
verborse = 1
validation_split = 0.2
lernrate = 0.001

loss = 'categorical_crossentropy'
metrics = ["accuracy"]

# Tensorflow -> tensortRT
path_tf_model = os.path.join(base_path, '../Models/Tensor-Model/tf_model')
path_h5_model = os.path.join(base_path, '../Models/Keras-Model/')
path_rt_opt_model = os.path.join(base_path, '/Models/RT-Model/rt_model')
path_to_frozen_model = os.path.join(base_path, '')

# config gpu
# tensorflow allocation memory
tf_all = 0.5

max_batch_size = 2
max_workspace_size_bytes = 2*(10**9)
# FP32 FP16 int8
precision_mode = "FP32"

# tensorflow model
meta_file_name = ''
output_model = ''
