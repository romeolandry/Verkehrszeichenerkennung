import tensorflow as tf
import config as cfg
import os
import numpy as np
from daten.Data_vorbebreitung import Data_vorbebreitung
from utils import *

base_path = os.getcwd()
pfad_to_test_data = os.path.join(base_path, "../Daten/Final_Training/Images")
pfad_to_tf_frozen_model = os.path.join(base_path, "../Models/Tensor-Model/tf_model_from_signs_model_Kalssifikation_2019_10_19_20_58_53/tf_frozen_from_signs_model_Kalssifikation_2019_10_19_20_58_53.pb")
pfad_to_trt_frozen_model = os.path.join(base_path, "../Models/RT-Model/trt_frozen_from_signs_model_Kalssifikation_2019_10_19_20_58_53.pb")


csv_beschreibung = cfg.path_to_class_beschreibung

sign_daten = Data_vorbebreitung(pfad_to_test_data,
                                cfg.IMG_SIZE,
                                cfg.path_to_class_beschreibung)
# Tensor("conv2d_input:0", shape=(?, 48, 48, 3), dtype=float32)
input_node = "conv2d_input:0"
# ensor("dense_1/Softmax:0", shape=(?, 43), dtype=float32)
output_node = "dense_1/Softmax:0"
print("Daten werden eingelesen.....")
test_image, match_list = sign_daten.load_image_test()
test_image = test_image[:10]
match_list = match_list[:10]
result_to_show = []

print("Inference tensorRt Model")

for test_data in zip(test_image, match_list):

    trt_prediction = perform_model(pfad_to_trt_frozen_model,
                                   input_node,
                                   output_node,
                                   test_data[0])
    predicted_label = sign_daten.get_roadsign_name(np.argmax(trt_prediction[0]))
    original_label = sign_daten.get_roadsign_name(np.argmax(test_data[1]))
    print("Ricttige Label ist :{}".format(original_label))
    print("TensorRT predictec {}".format(predicted_label))
    result_to_show.append((test_data[0],
                           original_label,
                           predicted_label,
                           trt_prediction[1]))
save_result_perform(result_to_show, "TensorRT Inference")

result_to_show = []
print("Inference tensorflow Model ")
for test_data in zip(test_image, match_list):

    tf_prediction = perform_model(pfad_to_tf_frozen_model,
                                  input_node,
                                  output_node,
                                  test_data[0])
    predicted_label = sign_daten.get_roadsign_name(np.argmax(tf_prediction[0]))
    original_label = sign_daten.get_roadsign_name(np.argmax(test_data[1]))
    print("Ricttige Label ist :{}".format(original_label))
    print("Tensorflow predictec {}".format(predicted_label))
    result_to_show.append((test_data[0],
                           original_label,
                           predicted_label,
                           tf_prediction[1]))
save_result_perform(result_to_show, "Tensorflow Inference")
