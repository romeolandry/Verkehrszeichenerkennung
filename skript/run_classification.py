import os
import sys
import datetime
import tensorflow as tf
from daten.Data_vorbebreitung import Data_vorbebreitung
from models.Classification import Classification, Classification_test
from tensorflow.python.keras import (losses,
                                     optimizers,
                                     metrics,
                                     regularizers)
from optparse import OptionParser

base_path = os.getcwd()
parser = OptionParser()

path_to_class_beschreibung = os.path.join(base_path, '../utils/' +
                                          'Text_Beschreibung.csv')
path_to_model = os.path.join(base_path, '../Models/Keras-Model/' +
                             'signs_model_Kalssifikation_{}.h5'.format(
                                 datetime.datetime.now().strftime(
                                     "%Y_%m_%d_%H_%M_%S")))
parser.add_option("-m", "--mode",
                  dest="run_mode",
                  help="run_classification fürs Trainen oder Test",
                  default="train")

parser.add_option("-p", "--path", dest="path_to_gtsrb",
                  help="pafd zum trainins- oder testsdataset")

parser.add_option("-d", "--desc",
                  dest="csv_beschreibung",
                  help="Textbeschreibung von Zeichnen",
                  default=path_to_class_beschreibung)

parser.add_option("-s", "--save",
                  dest="pafdsmodel",
                  help="pfad zur Speicherung des Models",
                  default=path_to_model)
parser.add_option("-l", "--load",
                  dest="path_weight",
                  help="Pfad zu das trainiete Model")

(options, args) = parser.parse_args()

# path_to_gtsrb = os.path.join(base_path, '../Daten/Final_Training/Images')
run_mode = options.run_mode
if not options.path_to_gtsrb:
    print("Pafd zum Dataset wird gebraucht!")
else:
    path_to_gtsrb = options.path_to_gtsrb

if (options.run_mode == "test"):
    if (not options.path_weight):
        print("Pafd zum weight wird benötigt!")
        exit()
    else:
        path_weight = options.path_weight

csv_beschreibung = options.csv_beschreibung
pafdsmodel = options.pafdsmodel


IMG_SIZE = 48
NUM_BATCH = 64
NUM_EPOCHS = 1000
verborse = 1
validation_split = 0.2
lernrate = 0.001

optimizer = optimizers.Adamax(lernrate)
loss = 'categorical_crossentropy'
metrics = ["accuracy"]
data_vorbereitung = Data_vorbebreitung(path_to_gtsrb, IMG_SIZE,
                                       path_to_class_beschreibung)

classes_count = 43

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
    train_model = Classification(path_to_model, IMG_SIZE, classes, loss,
                                 optimizer, metrics, verborse,
                                 validation_split, NUM_BATCH, NUM_EPOCHS)
    model = train_model.build_model()
    print("Training")
    train_model.train_model(model, train_images[100:], train_labels[100:])
else:
    # Test model
    print("#####################################")
    print("###### Testphase #############")
    print("##################################")
    print("Daten werden eingelesen.....")
    test_image, match_list = data_vorbereitung.load_image_test()
    # data_vorbereitung.display_roadsign_classes(test_image, 0)
    print("Daten eingelesen!")
    test_model = Classification_test(path_weight, IMG_SIZE, classes_count)

    # aplying des Models
    print("Anwendung des Modells")
    roadsign_images, predicted_classes = test_model.test_model(optimizer,
                                                               metrics, loss,
                                                               test_image)
    # framed_img = utilis.frame_image
    data_vorbereitung.display_prediction_vs_real_classes(roadsign_images,
                                                         match_list,
                                                         predicted_classes)
