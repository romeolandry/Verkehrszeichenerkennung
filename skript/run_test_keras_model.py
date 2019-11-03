import os
import sys
import tensorflow as tf
from daten.Data_vorbebreitung import Data_vorbebreitung
from models.Classification import Classification, Classification_test

from optparse import OptionParser
from utils import save_performance_model, loard_all_model_for_test
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

parser.add_option("-m", "--model",
                  dest="csv_for_model",
                  help="Pfad zu das trainiete Model",
                  default=None)

(options, args) = parser.parse_args()

csv_beschreibung = options.csv_beschreibung

csv_for_model = None

if not options.path_to_gtsrb:
    print("Pafd zum Dataset wird gebraucht!")
else:
    path_to_gtsrb = options.path_to_gtsrb

if options.csv_for_model is None:
    if not options.path_keras_saved_model:
        print("Give Pfad zum Model")
    else:
        path_keras_saved_model = options.path_keras_saved_model
else:
    csv_for_model = options.csv_for_model

csv_beschreibung = options.csv_beschreibung


data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, cfg.IMG_SIZE,
                                       csv_beschreibung)

# Test model
print("#####################################")
print("###### Testphase #############")
print("##################################")
print("Daten werden eingelesen.....")


test_image, match_list = data_vorbereitung.load_image_test()

print("Daten eingelesen!")
if csv_for_model is not None:
    list_name_pfad = loard_all_model_for_test(csv_for_model)
    for elt in list_name_pfad:
        test_model = Classification_test(elt[1],
                                         cfg.IMG_SIZE,
                                         cfg.classes_count)

        # aplying des Models
        print("Anwendung des Modells")
        roadsign_images, predicted_classes = test_model.test_model(test_image)
        # framed_img = utilis.frame_image
        data_vorbereitung.display_prediction_vs_real_classes(roadsign_images,
                                                             match_list,
                                                             predicted_classes,
                                                             elt[0])

else:
    test_model = Classification_test(path_keras_saved_model,
                                     cfg.IMG_SIZE,
                                     cfg.classes_count)

    # aplying des Models
    print("Anwendung des Modells")
    roadsign_images, predicted_classes = test_model.test_model(test_image)
    # framed_img = utilis.frame_image
    data_vorbereitung.display_prediction_vs_real_classes(roadsign_images,
                                                         match_list,
                                                         predicted_classes,
                                                         "keras_model_sign_vgg_ohne_gen")
