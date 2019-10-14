import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import (InputLayer, Dropout,
                                            BatchNormalization, MaxPooling2D,
                                            Conv2D, Flatten, Dense)


class Classification_test:
    def __init__(self, path_to_save, IMG_SIZE):
        self.__path_to_save = path_to_save
        self.__IMG_SIZE = IMG_SIZE

    def get_path_to_save(self):
        return self.__path_to_save

    def get_IMG_SIZE(self):
        return self.__IMG_SIZE

    def test_model(self, images):
        """
        test_model recieve a list of images: all images shall as be resize (-1,
        IMG_SIZE,IMG_SIZE,3)
        """
        #  Model
        model = load_model(self.__path_to_save)
        roadsign_images = []
        predicted_class = []
        for image in images:
            predicted_roadsign_class = np.argmax(model.predict(
                image.reshape(-1, self.__IMG_SIZE, self.__IMG_SIZE, 3)))
            roadsign_images.append(image)
            predicted_class.append(predicted_roadsign_class)
        return roadsign_images, predicted_class


class Classification(Classification_test):
    def __init__(self, path_to_save, IMG_SIZE, loss_function, optimizer,
                 metrics, verbose, validation_split,
                 output, batch_size, Num_epochs):
        super().__init__(path_to_save, IMG_SIZE)
        self.__loss_function = loss_function
        self.__optimizer = optimizer
        self.__metrics = metrics
        self.__verbose = verbose
        self.__validation_split = validation_split
        self.__output = output
        self.__batch_size = batch_size
        self.__Num_epochs = Num_epochs

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         padding='same', activation='relu',
                         input_shape=(super().get_IMG_SIZE(),
                         super().get_IMG_SIZE(), 3),
                         data_format="channels_last"))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.__output, activation='softmax'))
        model.summary()

        return model

    def train_model(self, model, train_images, train_labels):
        model.compile(loss=self.__loss_function, optimizer=self.__optimizer,
                      metrics=self.__metrics)
        print("______________Anfang des Trainings____________________")
        model.fit(train_images, train_labels, epochs=self.__Num_epochs,
                  batch_size=self.__batch_size, verbose=1,
                  validation_split=self.__validation_split)
        print("training fertig")
        print("saving...")
        model.save(str(super().get_path_to_save()) +
                   'signs_model_Kalssifikation.h5')
        print("saved")
