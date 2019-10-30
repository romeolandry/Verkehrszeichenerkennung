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

parser.add_option("-t", "--gen",
                  dest="data_gen",
                  help="train Model with data generator",
                  default=False)

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

# image und Labels zum Trainen werden gelesen
print("Daten werden eingelesen.....")
(train_images,
    train_labels,
    classes) = data_vorbereitung.load_roadsigns_data('all')
print("Daten eingelesen!")

print("#####################################")
print("###### Trainingsphase #############")
print("##################################")
print("Aufbau des Models!")
optimizer = optimizers.RMSprop()

print("train model :", cfg.keras_model_name)
if not options.data_gen:
    # ########################################################################################################
    print("#____________________________________simple training without gen!")
    cfg.keras_model_name = "keras_model_sign_simple_ohne_gen"
    save_model_to = os.path.join(pfad_des_model,
                                 cfg.keras_model_name + ".h5")
    train_model = Classification(save_model_to,
                                 cfg.IMG_SIZE,
                                 classes,
                                 cfg.loss,
                                 optimizer,
                                 cfg.metrics,
                                 cfg.verborse,
                                 cfg.validation_split,
                                 cfg.NUM_BATCH,
                                 cfg.NUM_EPOCHS)
    model = train_model.build_model()  # oder build_model build_vgg_model

    history = train_model.train_model(model,
                                      train_images,
                                      train_labels)
    print("##### max validation acc :", max(history.history['val_acc']))
    print("##### min val_loss :", min(history.history['val_loss']))

    save_performance_model(cfg.keras_model_name,
                           cfg.loss,
                           optimizer,
                           cfg.lernrate,
                           history)

    #########################################################################
    print("#____________________________________simple training with gen!")
    cfg.keras_model_name = "keras_model_sign_simple_gen"
    save_model_to = os.path.join(pfad_des_model,
                                 cfg.keras_model_name + ".h5")
    train_model = Classification(save_model_to,
                                 cfg.IMG_SIZE,
                                 classes,
                                 cfg.loss,
                                 optimizer,
                                 cfg.metrics,
                                 cfg.verborse,
                                 cfg.validation_split,
                                 cfg.NUM_BATCH,
                                 cfg.NUM_EPOCHS)

    model = train_model.build_model()  # oder build_model

    history = train_model.train_model_gen(model,
                                          train_images,
                                          train_labels)
    print("##### max validation acc :", max(history.history['val_acc']))
    print("##### min val_loss :", min(history.history['val_loss']))

    save_performance_model(cfg.keras_model_name,
                           cfg.loss,
                           optimizer,
                           cfg.lernrate,
                           history)
    ###################################################################
    print("#____________________________________vgg training without gen!")
    cfg.keras_model_name = "keras_model_sign_vgg_ohne_gen"
    save_model_to = os.path.join(pfad_des_model,
                                 cfg.keras_model_name + ".h5")
    train_model = Classification(save_model_to,
                                 cfg.IMG_SIZE,
                                 classes,
                                 cfg.loss,
                                 optimizer,
                                 cfg.metrics,
                                 cfg.verborse,
                                 cfg.validation_split,
                                 cfg.NUM_BATCH,
                                 cfg.NUM_EPOCHS)

    model = train_model.build_vgg_model()  # oder build_model

    history = train_model.train_model(model,
                                      train_images,
                                      train_labels)
    print("##### max validation acc :", max(history.history['val_acc']))
    print("##### min val_loss :", min(history.history['val_loss']))

    save_performance_model(cfg.keras_model_name,
                           cfg.loss,
                           optimizer,
                           cfg.lernrate,
                           history)
    ###############################################################
    print("#____________________________________vgg training with gen!")
    cfg.keras_model_name = "keras_model_sign_vgg_gen"
    save_model_to = os.path.join(pfad_des_model,
                                 cfg.keras_model_name + ".h5")
    train_model = Classification(save_model_to,
                                 cfg.IMG_SIZE,
                                 classes,
                                 cfg.loss,
                                 optimizer,
                                 cfg.metrics,
                                 cfg.verborse,
                                 cfg.validation_split,
                                 cfg.NUM_BATCH,
                                 cfg.NUM_EPOCHS)

    model = train_model.build_vgg_model()  # oder build_model

    history = train_model.train_model_gen(model,
                                          train_images,
                                          train_labels)
    print("##### max validation acc :", max(history.history['val_acc']))
    print("##### min val_loss :", min(history.history['val_loss']))

    save_performance_model(cfg.keras_model_name,
                           cfg.loss,
                           optimizer,
                           cfg.lernrate,
                           history)
else:
    print("train wih generator")
    exit()
    history = train_model.train_model_gen(model,
                                          train_images,
                                          train_labels)

    print("##### max validation acc :", max(history.history['val_acc']))
    print("##### min val_loss :", min(history.history['val_loss']))

    save_performance_model(cfg.keras_model_name,
                           cfg.loss,
                           optimizer,
                           lernrate,
                           history)
