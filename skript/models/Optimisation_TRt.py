import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import config as cfg
import os

from tensorflow.keras.models import load_model
from tensorflow.python.platform import gfile
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.compat.v1.train import Saver
from tensorflow.compat.v1.keras import backend
from tensorflow.python.tools import freeze_graph
from utils import *

base_path = os.getcwd()


class Optimisation_TRT:
    """
        from pfad des keras -model wird die Name extrahieren
        bsp. ".../model_test.h5 wird "model_test" extrahieren.
        tensorflow-model wird dann gennant tf_model_from_"model_test"
            ordner from Tensor-Model wird so aussehen:
                tf_model_from_model_test
                        |_variable
                        |_saved_model.pb
                        |_tf_frozen_from_"model_test.pb"
            und RT-Model wird so aussehen
                RT-Model
                    |_trt_frozen_from_"model_test.pb"
        """
    def __init__(self,
                 path_h5_model,
                 keras_model_name):
        self.__keras_model_name = keras_model_name
        self.__path_h5_model = path_h5_model
        self.__tf_model_from_keras = "tf_model_from_" + keras_model_name
        tf_model = "tf_frozen_from_" + keras_model_name + ".pb"
        self.__tf_frozen_model_name = tf_model
        self.__trt_frozen_name = "trt_frozen_from_" + keras_model_name + ".pb"

        path_tf_model = '../Models/Tensor-Model/' + self.__tf_model_from_keras
        self.__path_tf_model = os.path.join(base_path, path_tf_model)

        self.__path_tf_frozen_model = os.path.join(self.__path_tf_model,
                                                   self.__tf_frozen_model_name)

        path_trt_model = '../Models/RT-Model/' + self.__trt_frozen_name
        self.__path_trt_frozen_model = os.path.join(base_path, path_trt_model)

    def keras_to_tensor_model(self):
        """
        diese Funktion wird ein Keras-Model in ein Tensor-Model
        Umwandeln
        """
        """ da die Batchnormalisation-Parameter im Keras
        als trainable parameter gespeichert sind, diese muss als
        non_trainable zurückgesetzt.
        """
        tf.keras.backend.set_learning_phase(0)

        print("Keras-Model wird hochgeladen!")
        try:
            model = load_model(self.__path_h5_model)
        except FileNotFoundError:
            print("Kein h5 Datei wurde gefunden!")
            exit()

        print("Umwandlung vom Keras zu Tensor-Model..")
        # keras session wird als tf- Session gelesen!
        # input und Output des Model wird gespeichert zum spätere Einsatz

        sess = backend.get_session()
        print("Save Input und Ouput des Models")
        cfg.input_model["input"] = model.inputs[0]
        cfg.output_model["output"] = model.outputs[0]
        # die Graph der Tf-Session wird gespeichert
        print("Speicherung des Tensor-Graph")
        tf.saved_model.simple_save(sess,
                                   self.__path_tf_model,
                                   inputs=cfg.input_model,
                                   outputs=cfg.output_model)
        # Das Model wird Compilert/ gewichtet mit dem trainierten Gewichten
        print("Compilierung des Models!")
        freeze_graph.freeze_graph(None,
                                  None,
                                  None,
                                  None,
                                  model.outputs[0].op.name,
                                  None,
                                  None,
                                  self.__path_tf_frozen_model,
                                  False,
                                  "",
                                  input_saved_model_dir=self.__path_tf_model)
        print("#___________________________________________________#")
        print("_das Models wurde umgewandelt und " +
              "gespeichert unter :{} _#".format(self.__path_tf_model))
        print("#___________________________________________________#")

    def trt_model_von_frozen_graph(self):
        """
        Die Funktion wird tensorflow Frozen-Model lesen und daraus ein
        TensorRT-Model aufbauen
        input : frozen graph .pd datei
        output: tensorRt- Model
        """

        print("tensor Forzen-Model wird gelesen!")
        frozen_graph = lord_model_graph(self.__path_tf_frozen_model)
        print("TensorRT model ergestellt!")
        trt_graph = trt.create_inference_graph(
            # forzen model
            input_graph_def=frozen_graph,
            # output vom Model
            outputs=[cfg.output_model['output']],
            # kann ein integer sein
            max_batch_size=cfg.max_batch_size,
            # 2*(10**9)
            max_workspace_size_bytes=cfg.max_workspace_size_bytes,
            # "FP32"
            precision_mode=cfg.precision_mode
        )
        print("TensorRT model wird gespeichert!")
        save_graph(self.__path_trt_frozen_model, trt_graph)
        print("#___________________________________________________#")
        print("#_TensorRT model wurde gespeichert " +
              "unter der Name:{}" +
              " gespeichert _#".format(self.__trt_frozen_name))
        print("#___________________________________________________#")
        print("#___________________________________________________#")
        print("#_ Das verzeichnis ist :{}".format(self.__path_tf_frozen_model))
        print("#___________________________________________________#")

    def perfom_trt_model(self, input_mode, output_mode, input_img):
        print("Perform Trt-Model")
        Prin("Anfang!")
        prediction = perform_model(self.__path_trt_frozen_model,
                                   input_model,
                                   output_model,
                                   input_img)
        print("Fertig!")

    def perfom_tft_model(self, input_mode, output_mode, input_img):
        print("Perform Tensorflow-Model")
        Prin("Anfang!")
        prediction = perform_model(self.__path_tf_frozen_model,
                                   input_mode,
                                   output_mode,
                                   input_img)
        print("Fertig!")
