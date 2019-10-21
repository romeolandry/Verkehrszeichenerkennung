import tensorflow as tf
from tensorflow.python.platform import gfile


def save_graph(path, frozen_graph):
    """
     die Funktion hilf die .pb-Datei zu speichern
    """
    with gfile.FastGFile(path, "wb") as f:
        f.write(frozen_graph.SerializeToString())


def lord_model_graph(path):
    """
     die Funktion hilf die .pb-Datei zu lesen
    """
    tf.reset_default_graph()
    with tf.gfile.FastGFile(path, "rb") as f:
        graph = tf.GraphDef()
        graph.ParseFromString(f.read())
    return graph


def check_optimisation_change(path_frozen_graph_tf, path_frozen_graph_rt):
    """
    Diese Funktion stell den Unterschied zwischen tf_model
    und TensorRt graph
    """
    tf_frozen_graph = lord_model_graph(path_frozen_graph_tf)
    all_nodes = len([1 for n in tf_frozen_graph.node])
    print("Anzahl von Noden im Tensorflow Frozen-Graph:", all_nodes)

    trt_graph = lord_model_graph(path_frozen_graph_rt)
    trt_engine = [1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp']
    trt_engine_nodes = len(trt_engine)
    print("Anzahl von Engine-Noden im TensorRt Frozen-Graph:",
          trt_engine_nodes)
    all_trt_nodes = [1 for n in trt_graph.node]
    print("Anzahl von Noden im TensorRt Frozen-Graph:",
          all_trt_nodes)

    return(all_nodes, trt_engine_nodes, all_trt_nodes)


def perform_model(path_frozen_graph, input_mode, output_mode, input_img):
    """
    Diese funktion wird dazu helfen, irgenwelches Model(Tensorflow/TensorRt)
    anzuwenden
    """
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_option)) as sess:
            print("der Graph wird gelesen!")
            frozen_graph = lord_model_graph(path_frozen_graph)
            print("der wurde gelesen!")

            # Auswahl von input und output
            tf.import_graph_def(trt_graph, name='')
            input = sess.graph.get_tensor_by_name(input_mode)
            output = sess.graph.get_tensor_by_name(output_mode)

            total_time = 0
            n_time_inference = 50
            out_pred = sess.run(output, feed_dict={input: input_img})

            for i in range(n_time_inference):
                t1 = time.time()
                out_pred = sess.run(output, feed_dict={input: input_img})
                t2 = time.time()
                delta_time = t2 - t1
                total_time += delta_time
                print("gebrauchte Zeit - " + str(i) + ": ", delta_time)
            print("mittelre Zeit: {}".format(total_time/n_time_inference))
    return out_pred
