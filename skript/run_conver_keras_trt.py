import config as cfg
import os

from models.Optimisation_TRt import Optimisation_TRT
from optparse import OptionParser
from daten.Data_vorbebreitung import Data_vorbebreitung
from utils import check_optimisation_change


parser = OptionParser()

parser.add_option("-p", "--path", dest="pfad_keras_model",
                  help="pafd zum trainierten Kears-model")

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
        print("#______________________________________________#")
        print("andere Model als h5 wird nicht übertragen!")
        print("Geben sie bitte ein .h5 keras Model")
        print("#______________________________________________#")
        exit()
    else:
        keras_model_name = datei_name.split(".")[0]
        print("#______________________________________________#")
        print("#___________Wurzel von der Modelnamen__________#")
        print(datei_name.split(".")[0])
        print("#______________________________________________#")

rt_optimizer = Optimisation_TRT(pfad_keras_model,
                                keras_model_name)

# Keras-Model umwandelt
# diese Funktion wird tensorflow_frozen Model generiert
path_tf = rt_optimizer.keras_to_tensor_model()

# diese Funktion wird das Tensorflow-Model optimieren
path_trt = rt_optimizer.trt_model_von_frozen_graph()

print("Infrence von den Beiden Model tensorflow-Model gegen tensorRT-Model")
check_optimisation_change(path_tf, path_trt)

# read test Data zu testen
