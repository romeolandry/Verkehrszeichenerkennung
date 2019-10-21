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


class Optimisation_TRT:
    def __init__(self,
                 path_h5_model,
                 path_tf_model,
                 path_rt_opt_model,
                 path_to_frozen_model):

        self.__path_h5_model = path_h5_model
        self.__path_tf_model = path_tf_model
        self.__path_rt_opt_model = path_rt_opt_model
        self.__path_to_frozen_model = path_to_frozen_model

    def keras_to_tensor_model(self, train_config):
        """
        diese Funktion wird ein Keras-Model in ein Tensor-Model
        Umwandeln
        """
        """ da die Batchnormalisation-Parameter im Keras
        als trainable parameter gespeichert sind, mussen wird diese als
        non_trainable zurücksetzen
        """
        if train_config:
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
                                  os.path.join(self.__path_tf_model,
                                               cfg.tensor_frozen_name),
                                  False,
                                  "",
                                  input_saved_model_dir=self.__path_tf_model)
        print("das Models wurde umgewandelt und " +
              "gespeichert unter :{}".format(self.__path_tf_model))
        print("das Tensorflow-Frozen-Model wurde " +
              "unter der Name:{} gespeichert".format(cfg.tensor_frozen_name))

    """ def conver_tf_to_frozen_model(self, output_model):

         das aus Keras-Model Generierte Tensort-Model(oder ein normales
         Tensor-Model) wird hochgeladen
         und in frozen Graph umgewandelt
         input : tensor-Model
         ouput : frozen graph .pd

        # set the memory fraction eg. 0.2 meaning tf use 20% of gpu and
        # the will bei Trt
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.tf_all)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_option)) as sess:
            # import of meta graph fo tensorflow model
            path_meta_file = self.__path_tf_model + ".meta"
            saver = tf.train.import_meta_graph(path_meta_file)
            # print(graph)
            # exit()
            # restore the weights to the meta graph
            saver.restore(sess, self.__path_tf_model)

            # specify the tensor output of the model
            output_models = output_model
            # print(output_models)
            # exit()
            # convert to frozen model
            print("Frozen model ergestellt!")
            frozen_graph = convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names=output_models)

            # save the frozen graph
            print("speichert!")
            save_graph(self.__path_to_frozen_model, frozen_graph)
            print("Frozen model wurde gespeichert!")
 """
    def trt_model_von_frozen_graph(self):
        """
        Die Funktion wird frozen lesen und daraus ein TensorRT-Model
        aufbauen
        input : frozen graph .pd datei
        output: tensorRt- Model
        """

        print("tensor Forzen-Model wird gelesen!")
        frozen_pfad = os.path.join(self.__path_tf_model, "tf_frozen_model.pb")
        print(frozen_pfad)
        frozen_graph = lord_model_graph(frozen_pfad)
        print("TensorRT model ergestellt!")
        print(cfg.output_model['output'])
        # exit()
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
        save_graph("/home/kamgo/Donnees/Master_projekt/TensorRT/Verkehrszeichnerkennung/Models/RT-Model/test.pb", trt_graph)
        print("TensorRT model wurde gespeichert!")

    def perfom_trt_model(self, input_mode, output_mode, input_img):
        print("Perform Trt-Model")
        Prin("Anfang!")
        prediction = perform_model(self.__path_rt_opt_model,
                                   input_model,
                                   output_model,
                                   input_img)
        print("Fertig!")

    def perfom_tft_model(self, input_mode, output_mode, input_img):
        print("Perform Tensorflow-Model")
        Prin("Anfang!")
        prediction = perform_model(self.__path_rt_opt_model,
                                   input_mode,
                                   output_mode,
                                   input_img)
        print("Fertig!")
