import os
import sys
import tensorflow as tf
from daten.Data_vorbebreitung import Data_vorbebreitung
from models.Classification import Classification, Classification_test
from tensorflow.python.keras import (losses,
                                     optimizers,
                                     metrics,
                                     regularizers)
from optparse import OptionParser
from utils import save_performance_model
import config as cfg

parser = OptionParser()

parser.add_option("-p", "--path", dest="path_to_gtsrb",
                  help="pafd zum trainins- oder testsdataset")

parser.add_option("-d", "--desc",
                  dest="csv_beschreibung",
                  help="Textbeschreibung von Zeichnen",
                  default=cfg.path_to_class_beschreibung)

parser.add_option("-l", "--load",
                  dest="path_keras_saved_model",
                  help="Pfad zu das trainiete Model")

(options, args) = parser.parse_args()

csv_beschreibung = options.csv_beschreibung

if not options.path_to_gtsrb:
    print("Pafd zum Dataset wird gebraucht!")
else:
    path_to_gtsrb = options.path_to_gtsrb

csv_beschreibung = options.csv_beschreibung
path_keras_saved_model = options.path_keras_saved_model

data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, cfg.IMG_SIZE,
                                       csv_beschreibung)


# Test model
print("#####################################")
print("###### Testphase #############")
print("##################################")
print("Daten werden eingelesen.....")
optimizer = optimizers.Adamax(cfg.lernrate)
test_image, match_list = data_vorbereitung.load_image_test()
# data_vorbereitung.display_roadsign_classes(test_image, 0)
print("Daten eingelesen!")
test_model = Classification_test(path_keras_saved_model,
                                 cfg.IMG_SIZE,
                                 cfg.classes_count)

# aplying des Models
print("Anwendung des Modells")
roadsign_images, predicted_classes = test_model.test_model(optimizer,
                                                           cfg.metrics,
                                                           cfg.loss,
                                                           test_image)
# framed_img = utilis.frame_image
data_vorbereitung.display_prediction_vs_real_classes(roadsign_images,
                                                     match_list,
                                                     predicted_classes)
