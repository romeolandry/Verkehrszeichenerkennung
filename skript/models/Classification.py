import numpy as np
import os
from datetime import datetime
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import (InputLayer, Dropout,
                                            BatchNormalization, MaxPooling2D,
                                            Conv2D, Flatten, Dense)
from tensorflow.python.keras.callbacks import (EarlyStopping,
                                               ModelCheckpoint,
                                               TensorBoard)
from daten.data_augmentation import keras_data_gen
from sklearn.model_selection import train_test_split
import config as cfg


class Classification_test:
    def __init__(self, path_to_save, IMG_SIZE, output):
        self.__path_to_save = path_to_save
        self.__IMG_SIZE = IMG_SIZE
        self.__output = output

    def get_path_to_save(self):
        return self.__path_to_save

    def get_IMG_SIZE(self):
        return self.__IMG_SIZE

    def get_output(self):
        return self.__output

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         padding='same', activation='relu',
                         input_shape=(self.__IMG_SIZE,
                         self.__IMG_SIZE, 3),
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

    def compile_model(self, model):
        raise NotImplementedError("muss die Methde in" +
                                  "ClassificationsKlasse überschreiben!")

    def test_model(self, optimizer, metric, loss, images):
        """
        test_model recieve a list of images: all images shall as be resize (-1,
        IMG_SIZE,IMG_SIZE,3)
        """
        #  Model build
        print("das Model wird aufgebaut und compile")
        model = self.build_model()
        model.load_weights(self.__path_to_save)
        """ model.compile(loss=loss, optimizer=optimizer,
                      metrics=metric) """
        roadsign_images = []
        predicted_class = []
        print("prediction")
        for image in images:
            predicted_roadsign_class = np.argmax(model.predict(
                image.reshape(-1, self.__IMG_SIZE, self.__IMG_SIZE, 3)))
            roadsign_images.append(image)
            predicted_class.append(predicted_roadsign_class)
        print("End of Prediction!")
        return roadsign_images, predicted_class


class Classification(Classification_test):
    def __init__(self, path_to_save, IMG_SIZE, output,
                 loss_function, optimizer,
                 metrics, verbose, validation_split,
                 batch_size, Num_epochs):
        super().__init__(path_to_save, IMG_SIZE, output)
        self.__loss_function = loss_function
        self.__optimizer = optimizer
        self.__metrics = metrics
        self.__verbose = verbose
        self.__validation_split = validation_split
        self.__batch_size = batch_size
        self.__Num_epochs = Num_epochs

    def build_model(self):
        return super().build_model()

    def compile_model(self, model):
        model.compile(loss=self.__loss_function, optimizer=self.__optimizer,
                      metrics=self.__metrics)

    def train_model(self, model, train_images, train_labels):
        # Define the Keras TensorBoard callback.
        # logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=logdir)
        # Early stopping complex
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=cfg.patience)
        # modelcheckpoint
        mc = ModelCheckpoint(super().get_path_to_save(),
                             monitor='val_acc',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
        # tensorborad
        logdir = os.path.join(cfg.pfad_zu_logs, cfg.keras_model_name)
        tb = TensorBoard(log_dir=logdir,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=False,)
        callbak = [es, mc, tb]
        self.compile_model(model)
        print("______________Anfang des Trainings____________________")
        history = model.fit(train_images,
                            train_labels,
                            epochs=self.__Num_epochs,
                            batch_size=self.__batch_size,
                            verbose=1,
                            validation_split=self.__validation_split,
                            callbacks=callbak)
        print("training fertig")
        return history

    def train_model_with_data_generator(self,
                                        model,
                                        train_images,
                                        train_labels):
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=cfg.patience,
                           min_delta=cfg.min_delta)
        # modelcheckpoint
        mc = ModelCheckpoint(super().get_path_to_save(),
                             monitor='val_acc',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
        # tensorborad
        logdir = os.path.join(cfg.pfad_zu_logs, cfg.keras_model_name)
        tb = TensorBoard(log_dir=logdir,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=False,)
        callbak = [es, mc, tb]

        self.compile_model(model)
        # convert input list (images and labels) to numpy array
        images = np.array(train_images)
        labels = np.array(train_labels)

        print("shape train Image: ", images.shape)
        print("shape train label: ", labels.shape)
        # split train data to 75% for train and 25% for validation

        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=cfg.validation_split,
            stratify=None)

        # generate images for training
        train_datagen, val_datagen = keras_data_gen()
        train_generator = train_datagen.flow(train_images,
                                             train_labels,
                                             batch_size=self.__batch_size)
        # rescale image validation
        val_generator = val_datagen.flow(val_images,
                                         val_labels,
                                         batch_size=self.__batch_size)
        history = model.fit_generator(
            train_generator,
            epochs=self.__Num_epochs,
            steps_per_epoch=len(train_images)/self.__batch_size,
            validation_data=val_generator,
            validation_steps=len(train_labels)/self.__batch_size,
            callbacks=callbak)
        return history
