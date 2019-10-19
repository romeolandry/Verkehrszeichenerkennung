import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from tensorflow.keras.models import load_model
from tensorflow.python.platform import gfile
from utils import *


class Optimisation_TRT:
    def __init__(self, __path_tf_model, path_h5_model,
                 path_rt_opt_model, path_to_frozen_model):
        self.__path_tf_model = __path_tf_model
        self.__path_h5_model = path_h5_model
        self.__path_rt_opt_model = path_rt_opt_model
        self.__path_to_frozen_model = path_to_frozen_model

    def keras_to_tensor_model(self):
        """
        diese Funktion wird ein Keras-Model in ein Tensor-Model
        Umwandeln
        """
        print("Keras-Model wird hochgeladen!")
        model = load_model(self.__path_h5_model)
        print("Umwandlung vom Keras zu Tensor-Model..")
        saver = tf.train.saver()
        # keras session wird als tf- Session gelesen!
        sess = tf.keras.backend.get_session()
        # die Graph der Tf-Session wird gespeichert
        save_path = saver.save(sess, self.__path_tf_model)
        print("Umgewandelt und gespeichert!")

    def conver_tf_to_frozen_model(self, meta_file_name, output_model):
        """
         das aus Keras-Model Generierte Tensort-Model(oder ein normales
         Tensor-Model) wird hochgeladen
         und in frozen Graph umgewandelt
         input : tensor-Model
         ouput : frozen graph .pd
        """
        # set the memory fraction eg. 0.2 meaning tf use 20% of gpu and
        # the will bei Trt
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_option)) as sess:
            # import of meta graph fo tensorflow model
            path_meta_file = self.__path_tf_model + meta_file_name
            saver = tf.train.import_meta_graph(path_meta_file)
            # restore the weights to the meta graph
            saver.restore(sess, self.__path_tf_model)

            # specify the tensor output of the model
            output_models = [output_model]

            # convert to frozen model
            print("Frozen model ergestellt!")
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names=output_models
            )

            # save the frozen graph
            print("speichert!")
            save_graph(self.__path_to_frozen_model)
            print("Frozen model wurde gespeichert!")

    def trt_model_von_frozen_graph(self,
                                   output_model,
                                   max_batch_size,
                                   max_workspace_size_bytes,
                                   precision_mode):
        """
        Die Funktion wird frozen lesen und daraus ein TensorRT-Model
        aufbauen
        input : frozen graph .pd datei
        output: tensorRt- Model
        """

        print("Liest forzen model!")
        frozen_graph = lord_model_graph(self.__path_to_frozen_model)
        print("TensorRT model ergestellt!")
        trt_graph = trt.create_inference_graph(
            # forzen model
            input_graph_def=frozen_graph,
            # output vom Model
            outputs=[output_model],
            # kann ein integer sein
            max_batch_size=max_batch_size,
            # 2*(10**9)
            max_workspace_size_bytes=max_workspace_size_bytes,
            # "FP32"
            precision_mode=precision_mode
        )
        print("TensorRT model wird gespeichert!")
        save_graph(self.__path_rt_opt_model)
        print("TensorRT model wurde gespeichert!")

    def perfom_trt_model(self, input_mode, output_mode, input_img):
        print("Perform Trt-Model")
        Prin("Anfang!")
        prediction = perform_model(self.__path_rt_opt_model,
                                   input_mode,
                                   output_mode,
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
