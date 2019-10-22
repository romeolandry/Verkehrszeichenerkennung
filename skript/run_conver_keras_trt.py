import config as cfg
import os

from models.Optimisation_TRt import Optimisation_TRT
from optparse import OptionParser


parser = OptionParser()

parser.add_option("-p", "--path", dest="pfad_keras_model",
                  help="pafd zum trainierten Kears-model")

parser.add_option("-d", "--dest",
                  dest="path_trt_opt_model",
                  help="ordner von TensorRT-model",
                  default=cfg.trt_frozen_name)

(options, args) = parser.parse_args()

if (not options.pfad_keras_model):
    print("sie mussen ein gültige Keras-Model als Argument eingeben!")
    exit()
else:
    print("Datei-Name wird von dem Pfad gelesen!")
    pfad_keras_model = options.pfad_keras_model
    pfad_dir, datei_name = os.path.split(pfad_keras_model)
    print("Conten Ordner:{}".format(pfad_dir))
    print("file name {}".format(datei_name))
    print("keras wird geprüft!")
    if (datei_name.split(".")[1] != "h5"):
        print("andere Model als h5 wird übertragen!")
    else:
        print("Name:{}".format(name))
        print("extension {}".format(ext))
        cfg.keras_model_name = datei_name.split(".")[0]

path_trt_opt_model = options.path_trt_opt_model

rt_optimizer = Optimisation_TRT(pfad_keras_model,
                                path_trt_opt_model)
# Keras-Model umwandelt
rt_optimizer.keras_to_tensor_model()

# optimierung des Models
# 1- tf_model -> frozen_model.pb
# rt_optimizer.conver_tf_to_frozen_model(output_model)

rt_optimizer.trt_model_von_frozen_graph()
