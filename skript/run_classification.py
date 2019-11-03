import os
import sys
import tensorflow as tf
from daten.Data_vorbebreitung import Data_vorbebreitung
from models.Classification import Classification, Classification_test
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

# image und Labels zum Trainen werden gelesen
print("Daten werden eingelesen.....")
(train_images,
    train_labels,
    val_images,
    val_labels,
    classes) = data_vorbereitung.load_roadsigns_data(200)
print("Daten eingelesen!")
print("#####################################")
print("###### Trainingsphase #############")
print("##################################")
print("Aufbau des Models!")

print("train model :", cfg.keras_model_name)

if cfg.data_gen:
    if cfg.simple_model:
        print("Training des Simple_Model mit Data-Generator !")
        cfg.keras_model_name = "keras_model_sign_simple_gen_3"
        save_model_to = os.path.join(pfad_des_model,
                                     cfg.keras_model_name + ".h5")
        train_model = Classification(save_model_to,
                                     cfg.IMG_SIZE,
                                     classes,
                                     cfg.loss,
                                     cfg.optimizer,
                                     cfg.metrics,
                                     cfg.verborse,
                                     cfg.validation_split,
                                     cfg.NUM_BATCH,
                                     cfg.NUM_EPOCHS)

        model = train_model.build_model()

        history = train_model.train_model_gen(model,
                                              train_images,
                                              train_labels,
                                              val_images,
                                              val_labels)
        print("##### max validation acc :", max(history.history['val_acc']))
        print("##### min val_loss :", min(history.history['val_loss']))

        save_performance_model(cfg.keras_model_name,
                               cfg.loss,
                               cfg.optimizer,
                               cfg.lernrate,
                               history)
    if cfg.simple_vgg:
        print("Training des Vgg-Models mit Data-generator!")
        cfg.keras_model_name = "keras_model_sign_vgg_gen"
        save_model_to = os.path.join(pfad_des_model,
                                     cfg.keras_model_name + ".h5")
        train_model = Classification(save_model_to,
                                     cfg.IMG_SIZE,
                                     classes,
                                     cfg.loss,
                                     cfg.optimizer,
                                     cfg.metrics,
                                     cfg.verborse,
                                     cfg.validation_split,
                                     cfg.NUM_BATCH,
                                     cfg.NUM_EPOCHS)

        model = train_model.build_vgg_model()  # oder build_model

        history = train_model.train_model_gen(model,
                                              train_images,
                                              train_labels,
                                              val_images,
                                              val_labels)
        print("##### max validation acc :", max(history.history['val_acc']))
        print("##### min val_loss :", min(history.history['val_loss']))

        save_performance_model(cfg.keras_model_name,
                               cfg.loss,
                               cfg.optimizer,
                               cfg.lernrate,
                               history)
if cfg.ohne_gen:
    if cfg.simple_model:
        print("Training des Simple_Model ohne Data-Generator !")
        cfg.keras_model_name = "keras_model_sign_simple_ohne_data"
        save_model_to = os.path.join(pfad_des_model,
                                     cfg.keras_model_name + ".h5")
        train_model = Classification(save_model_to,
                                     cfg.IMG_SIZE,
                                     classes,
                                     cfg.loss,
                                     cfg.optimizer,
                                     cfg.metrics,
                                     cfg.verborse,
                                     cfg.validation_split,
                                     cfg.NUM_BATCH,
                                     cfg.NUM_EPOCHS)
        model = train_model.build_model()  # oder build_model build_vgg_model

        history = train_model.train_model(model,
                                          train_images,
                                          train_labels,
                                          val_images,
                                          val_labels)

        print("##### max validation acc :", max(history.history['val_acc']))
        print("##### min val_loss :", min(history.history['val_loss']))

        save_performance_model(cfg.keras_model_name,
                               cfg.loss,
                               cfg.optimizer,
                               cfg.lernrate,
                               history)
    if cfg.simple_vgg:
        print(" Training des Vgg-Models ohne Data-generator!")
        cfg.keras_model_name = "keras_model_sign_vgg_ohne_gen"
        save_model_to = os.path.join(pfad_des_model,
                                     cfg.keras_model_name + ".h5")
        train_model = Classification(save_model_to,
                                     cfg.IMG_SIZE,
                                     classes,
                                     cfg.loss,
                                     cfg.optimizer,
                                     cfg.metrics,
                                     cfg.verborse,
                                     cfg.validation_split,
                                     cfg.NUM_BATCH,
                                     cfg.NUM_EPOCHS)

        model = train_model.build_vgg_model()

        history = train_model.train_model(model,
                                          train_images,
                                          train_labels,
                                          val_images,
                                          val_labels)
        print("##### max validation acc :", max(history.history['val_acc']))
        print("##### min val_loss :", min(history.history['val_loss']))

        save_performance_model(cfg.keras_model_name,
                               cfg.loss,
                               cfg.optimizer,
                               cfg.lernrate,
                               history)
