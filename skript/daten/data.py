import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import PIL

from PIL import Image
from pandas import read_csv
from skimage import color, exposure, transform, io
from tensorflow.python.keras.utils import np_utils
from textwrap import wrap


class Data_vorbebreitung:
    def __init__(self, path_to_gtsrb, IMG_SIZE, path_to_description):
        self.__path_to_gtsrb = path_to_gtsrb
        self.__IMG_SIZE = IMG_SIZE
        self.__path_to_description = path_to_description
        self.__sign_names = []

    def read_traffic(self):
        """
        Argument: die  Funktion nimmt als eingebe den Pfad zum Data,
        ZB.  Data/Final_Trainnig/images
        Returns : list von Bildern und entsprechende Labels
        """
        images = []
        labels = []
        for c in range(0, 43):
            #  build the path to subdirectory eg. 00000 00001
            prefix = self.__path_to_gtsrb + '/' + format(c, '05d') + '/'
            file_path = prefix + 'GT-' + format(c, '05d') + '.csv'
            with open(file_path, 'r') as gt_file:
                gt_reader = csv.reader(gt_file, delimiter=';')
                #  header skipt of csv-files
                next(gt_reader)
                for row in gt_reader:
                    images.append(plt.imread(prefix + row[0]))
                    labels.append(row[7])
        return images, labels

    def prepocess_img(self, img):
        """
        Da die bilder nicht die gleiche große haben,
        muss jedes erstmal zur gleichen göße dimensionieren
        """
        img = np.asarray(img)
        img = transform.resize(img, (self.__IMG_SIZE, self.__IMG_SIZE))
        return img

    def load_roadsigns_data(self):
        """
        Argument: die  Funktion nimmt als eingebe den Pfad zum Data,
        ZB.  Data/Final_Trainnig/images
        und Pfad zur Beschreibung des labels
        Returns : numay array  von Bildern und entsprechende Labels
        und set die Variable sign_names
        """
        # read csv_Datei zur sign_names
        with open(self.__path_to_description, 'r', newline='') as desc:
            desc_reader = csv.reader(desc, delimiter=',')
            next(desc_reader)
            for row in desc_reader:
                self.__sign_names.append(row)
        # read Images
        images = []
        labels = []
        num_raodsign_classes = len([pic for pic in os.listdir(
            self.__path_to_gtsrb) if not pic.startswith(".")])
        for c in range(0, num_raodsign_classes):
            # build the path to subdirectory eg. 00000 00001
            prefix = self.__path_to_gtsrb + '/' + format(c, '05d') + '/'
            file_path = prefix + 'GT-' + format(c, '05d') + '.csv'
            with open(file_path, 'r', newline='') as gt_file:
                gt_reader = csv.reader(gt_file, delimiter=';')
                next(gt_reader)  # header skipt of csv-file s
                for row in gt_reader:
                    # das erste Bild von jedem orden wird immer angezeigt.
                    jpg_file = Image.open(prefix + row[0])
                    # skalierte das Bild
                    gs_image = self.prepocess_img(jpg_file)
                    images.append(gs_image)
                    labels.append(row[7])

        images = np.array(images)
        labels = np_utils.to_categorical(labels,
                                         num_classes=num_raodsign_classes)
        return images, labels

    def get_roadsign_name(self, index):
        return self.__sign_names[index][1]

    def display_roadsign_classes(self, images, max_image):
        plt.rc('font', size=6)
        plt.rcParams["figure.figsize"] = (10, 5)  # (10, 5) (with, heigth)
        fig, axarr = plt.subplots(6, 8)
        num = 0
        for i in range(0, 6):
            for p in range(0, 8):
                axarr[i][p].axis('off')
                if (num < max_image):
                    axarr[i, p].imshow(images[num], interpolation='nearest')
                    roadsign_name = "\n".join(wrap(self.get_roadsign_name(num),
                                                   15))
                    axarr[i, p].set_title("[" + str(num) + "]\n" +
                                          roadsign_name)
                    num += 1
        fig.suptitle('German Traffic Sign recongnisation Benchmark',
                     fontsize=16, fontweight="bold", y=0.1)
        plt.subplots_adjust(hspace=1)
        plt.show()


dt_vb = Data_vorbebreitung("/home/kamgo/Donnees/Master_projekt/TensorRT" +
                           "/Verkehrszeichnerkennung/Daten/" +
                           "Final_Training/Images", 45,
                           "/home/kamgo/Donnees/" +
                           "Master_projekt/TensorRT/Verkehrszeichnerkennung/" +
                           "utils/Text_Beschreibung.csv")
# img, lab = dt_vb.read_traffic()
img, labels = dt_vb.load_roadsigns_data()
dt_vb.display_roadsign_classes(img, 10)
# print(img[0])
# print("label of this image {}".format(labels[0]))
