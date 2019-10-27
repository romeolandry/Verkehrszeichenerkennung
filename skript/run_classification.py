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
from utils import save_performance_model, plot_performance_models
import config as cfg

parser = OptionParser()

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

(options, args) = parser.parse_args()


pfad_des_model = options.pfad_des_model
csv_beschreibung = options.csv_beschreibung

if not options.path_to_gtsrb:
    print("Pafd zum Dataset wird gebraucht!")
else:
    path_to_gtsrb = options.path_to_gtsrb


csv_beschreibung = options.csv_beschreibung
pfad_des_model = options.pfad_des_model

data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, cfg.IMG_SIZE,
                                       csv_beschreibung)


print("#####################################")
print("###### Trainingsphase #############")
print("##################################")

losses = ['categorical_crossentropy',
          'kullback_leibler_divergence']
lernrates = [1e-3, 1e-4, 1e-5, 1e-6]
count = 0

# image und Labels zum Trainen werden gelesen
print("Daten werden eingelesen.....")
(train_images,
    train_labels,
    classes) = data_vorbereitung.load_roadsigns_data(1)
print("Daten eingelesen!")
# data_vorbereitung.display_roadsign_classes(train_images, 43)
for loss in losses:
    for lernrate in lernrates:
        # Vorbereitung des Models
        print("Aufbau des Models!")
        cfg.keras_model_name = "keras_model_{}".format(count)
        save_model_to = os.path.join(
            pfad_des_model,
            cfg.keras_model_name + ".h5")

        optimizer = optimizers.Adamax(cfg.lernrate)
        train_model = Classification(save_model_to,
                                     cfg.IMG_SIZE,
                                     classes,
                                     loss,
                                     optimizer,
                                     cfg.metrics,
                                     cfg.verborse,
                                     cfg.validation_split,
                                     cfg.NUM_BATCH,
                                     cfg.NUM_EPOCHS)

        model = train_model.build_model()
        print("train model :", cfg.keras_model_name)

        """ history = train_model.train_model(model,
                                                train_images,
                                                train_labels) """

        history = train_model.train_model_with_data_generator(
                    model,
                    train_images,
                    train_labels)

        print("##### max validation acc :", min(history.history['val_acc']))
        print("##### min val_loss :", max(history.history['val_loss']))

        save_performance_model(cfg.keras_model_name + str(count),
                               loss,
                               lernrate,
                               max(history.history['val_acc']),
                               min(history.history['val_loss']))
        count += 1


print("save resume of validation and losses")
plot_performance_models(cfg.pfad_zu_performance_model)
