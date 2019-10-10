import os
import csv
import matplotlib.pyplot as plt

class Data_vorbebreitung:

    def __init__(self,path_to_gtsrb):
        self.__path_to_gtsrb = path_to_gtsrb
    
    def read_traffic(self):
        """
        Argument: die  Funktion nimmt als eingebe den Pfad zum Data, ZB.  Data/Final_Trainnig/images
        Returns : list von Bildern und entsprechende Labels
        """
        images = []
        labels =[]
        for c in range(0,43):
            prefix = self.__path_to_gtsrb + '/' + format(c,'05d') + '/' # build the path to subdirectory eg. 00000 00001
            with open(prefix + 'GT-' + format(c,'05d') + '.csv', 'r', newline='') as gt_file:
                gt_reader = csv.reader(gt_file,delimiter=';')
                headers = next(gt_reader) # header skipt of csv-file s
                for row in gt_reader:
                    images.append(plt.imread(prefix + row[0]))
                    labels.append(row[7])           
        return images,labels
        
dt_vb = Data_vorbebreitung("/home/kamgo/Donnees/Master_projekt/TensorRT/Daten/Final_Training/Images")
img, lab = dt_vb.read_traffic()
print(img[0])
print("label of this image {}".format(lab[0]))