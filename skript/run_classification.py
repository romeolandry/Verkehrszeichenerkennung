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
import config as cfg

parser = OptionParser()

parser.add_option("-m", "--mode",
                  dest="run_mode",
                  help="run_classification fürs Trainen oder Test",
                  default="train")

parser.add_option("-p", "--path", dest="path_to_gtsrb",
                  help="pafd zum trainins- oder testsdataset")

parser.add_option("-d", "--desc",
                  dest="csv_beschreibung",
                  help="Textbeschreibung von Zeichnen",
                  default=cfg.path_to_class_beschreibung)

parser.add_option("-s", "--save",
                  dest="pfad_des_model",
                  help="pfad zur Speicherung des Models",
                  default=cfg.path_to_model)
parser.add_option("-l", "--load",
                  dest="path_keras_saved_model",
                  help="Pfad zu das trainiete Model")

(options, args) = parser.parse_args()

run_mode = options.run_mode
pfad_des_model = options.pfad_des_model
csv_beschreibung = options.csv_beschreibung

if not options.path_to_gtsrb:
    print("Pafd zum Dataset wird gebraucht!")
else:
    path_to_gtsrb = options.path_to_gtsrb

if (options.run_mode == "test"):
    if (not options.path_keras_saved_model):
        print("Pafd zum weight wird benötigt!")
        exit()
    else:
        path_keras_saved_model = options.path_keras_saved_model

csv_beschreibung = options.csv_beschreibung
pfad_des_model = options.pfad_des_model

optimizer = optimizers.Adamax(cfg.lernrate)
data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, cfg.IMG_SIZE,
                                       csv_beschreibung)

if run_mode == 'train':
    print("#####################################")
    print("###### Trainingsphase #############")
    print("##################################")
    # image und Labels zum Trainen werden gelesen
    print("Daten werden eingelesen.....")
    (train_images,
        train_labels, classes) = data_vorbereitung.load_roadsigns_data()
    print("Daten eingelesen!")
    # data_vorbereitung.display_roadsign_classes(train_images, 43)
    # Vorbereitung des Models
    print("Aufbau des Models!")
    train_model = Classification(pfad_des_model, cfg.IMG_SIZE, classes,
                                 cfg.loss,
                                 optimizer,
                                 cfg.metrics,
                                 cfg.verborse,
                                 cfg.validation_split,
                                 cfg.NUM_BATCH,
                                 cfg.NUM_EPOCHS)

    model = train_model.build_model()
    print("Training")
    # train_model.train_model(model, train_images[100:], train_labels[100:]) 
    train_model.train_model_with_data_generator(model,
                                                train_images[0:10],
                                                train_labels[0:10])
else:
    # Test model
    print("#####################################")
    print("###### Testphase #############")
    print("##################################")
    print("Daten werden eingelesen.....")
    test_image, match_list = data_vorbereitung.load_image_test()
    # data_vorbereitung.display_roadsign_classes(test_image, 0)
    print("Daten eingelesen!")
    test_model = Classification_test(path_keras_saved_model,
                                     cgf.IMG_SIZE,
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
