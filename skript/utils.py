import tensorflow as tf
from tensorflow.python.platform import gfile
import config as cfg
import time
import matplotlib.pyplot as plt
import PIL
import numpy as np
from skimage import color, exposure, transform, io
import pandas as pd

import os
import pathlib


def spilt_image(image, label):
    val_size = int(round(cfg.validation_split*len(image)))
    train_image = image[val_size:]
    val_image = image[:val_size]
    train_label = label[val_size:]
    val_label = label[:val_size]
    return train_image, train_label, val_image, val_label


def save_graph(path, frozen_graph):
    """
     die Funktion hilf die .pb-Datei zu speichern
    """
    with gfile.FastGFile(path, "wb") as f:
        f.write(frozen_graph.SerializeToString())


def save_graph_not_existed(folder_path, file_name, frozen_graph):
    tf.compat.v1.io.write_graph(frozen_graph,
                                folder_path,
                                file_name,
                                as_text=False)


def loard_model_graph(path):
    """
     die Funktion hilf die .pb-Datei zu lesen
    """
    # tf.reset_default_graph()
    with tf.gfile.FastGFile(path, "rb") as f:
        graph = tf.GraphDef()
        graph.ParseFromString(f.read())
    return graph


def check_optimisation_change(path_frozen_graph_tf, path_frozen_graph_rt):
    """
    Diese Funktion stell den Unterschied zwischen tensorflow-Graph
    und TensorRt-Graph
    """
    tf_frozen_graph = loard_model_graph(path_frozen_graph_tf)
    all_nodes = len([1 for n in tf_frozen_graph.node])
    print("Anzahl von Noden im Tensorflow Frozen-Graph:", all_nodes)

    trt_graph = loard_model_graph(path_frozen_graph_rt)
    trt_engine = [1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp']
    trt_engine_nodes = len(trt_engine)
    print("Anzahl von Engine-Noden im TensorRt Frozen-Graph:",
          trt_engine_nodes)
    all_trt_nodes = [1 for n in trt_graph.node]
    print("Anzahl von Noden im TensorRt Frozen-Graph:",
          all_trt_nodes)

    return(all_nodes, trt_engine_nodes, all_trt_nodes)


def perform_model(path_frozen_graph, input_node, output_node, input_img):
    """
    Diese funktion wird dazu helfen, irgenwelches Model(Tensorflow/TensorRt)
    anzuwenden
    """
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_option)) as sess:
            print("der Graph wird gelesen!")
            frozen_graph = loard_model_graph(path_frozen_graph)
            print("der wurde gelesen!")

            # Auswahl von input und output
            tf.import_graph_def(frozen_graph, name='')
            input_tensor = sess.graph.get_tensor_by_name(input_node)
            output_tensor = sess.graph.get_tensor_by_name(output_node)

            total_time = 0
            n_time_inference = 50
            input_img = input_img.reshape(-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 3)
            out_pred = sess.run(output_tensor,
                                feed_dict={input_tensor: input_img})

            for i in range(n_time_inference):
                t1 = time.time()
                out_pred = sess.run(output_tensor,
                                    feed_dict={input_tensor: input_img})
                t2 = time.time()
                delta_time = t2 - t1
                total_time += delta_time
                print("gebrauchte Zeit - " + str(i) + ": ", delta_time)
            print("mittelre Zeit: {}".format(total_time/n_time_inference))
            mittelre_zeit = total_time/n_time_inference
    return out_pred, mittelre_zeit


def frame_image(img, color_name):

    if(color_name == "green"):
        color = (0.0, 125.0, 0.0)
    else:
        if(color_name == "red"):
            color = (125.0, 0.0, 0.0)

    border_size = 5
    ny, nx = img.shape[0], img.shape[1]

    if img.ndim == 3:
        framed_img = np.full((border_size+ny+border_size,
                              border_size+nx+border_size, img.shape[2]),
                             color)
    framed_img[border_size:-border_size, border_size:-border_size] = img

    return framed_img


def save_result_perform(result_perform, titel):

    plt.rc('font', size=12)
    plt.rcParams["figure.figsize"] = (10, 10)  # (10, 5) (with, heigth)
    fig, axarr = plt.subplots(1, 10, squeeze=False)
    num = 0
    for i in range(0, len(result_perform)):
        real_label = result_perform[i][1]
        predicted_labe = result_perform[i][2]

        color = "red"
        if(real_label == predicted_labe):
            color = "green"
        real_label = real_label.replace(" ", "\n")
        predicted_labe = predicted_labe.replace(" ", "\n")
        axarr[0][i].text(0, 60, "\n Real:\n" + real_label +
                         "\n\nErkannt :\n" + predicted_labe +
                         "\nim " + str(result_perform[i][3]),
                         color=color, verticalalignment='top',
                         horizontalalignment='left')
        image = frame_image(result_perform[i][0], color)
        axarr[0][i].imshow(image)
        axarr[0][i].axis('off')
    fig.suptitle('Ergebnisse ' + str(titel), fontsize=16, fontweight="bold")
    plt.savefig(cfg.pfad_to_ergebnis_bild + titel + '.png')
    plt.show()


def read_beschreibung():
    df = pd.read_csv(cfg.path_to_class_beschreibung, skiprows=1)
    print(df.to_dict())
    return df.to_dict()


def save_performance_model(model_name,
                           loss,
                           optimizer,
                           lernrate,
                           history):
    dict = {'name': [model_name],
            'loss': [loss],
            'Optimozer': [optimizer],
            'lernrate': [lernrate],
            'val_acc': [max(history.history['val_acc'])],
            'val_loss': [min(history.history['val_loss'])],
            'train_acc': [max(history.history['acc'])],
            'train_loss': [min(history.history['loss'])]}
    df = pd.DataFrame.from_dict(dict)       
    # check if file exist
    if not pathlib.Path(cfg.pfad_zu_performance_model).exists():
        print("create new file for performance")
        os.mknod(cfg.pfad_zu_performance_model)
        df.to_csv(cfg.pfad_zu_performance_model, sep=';', encoding='utf-8')
    else:
        print("performance wird update!")
        df.to_csv(cfg.pfad_zu_performance_model,
                  sep=';',
                  mode='a',
                  header=False)

        print("updated!")


def plot_performance_models(path):
    print("read for plot")
    df = pd.read_csv(path, sep=';')
    df.groupby(['name', 'val_acc']).size().unstack().plot(kind='bar',
                                                          stacked=True)
    plt.title("validation accuracy of all Model")
    plt.savefig(cfg.pfad_to_ergebnis_bild + "validations.png")

    df.groupby(['name', 'val_loss']).size().unstack().plot(kind='bar',
                                                           stacked=True)

    plt.title("validation losses of all Model")
    plt.savefig(cfg.pfad_to_ergebnis_bild + "losses.png")


def loard_all_model_for_test(csv_pfad_performance):
    """ Input Csv
        out_put= list tupel (model_name,path_zu_model)
    """
    list_model_name = []
    with open(csv_pfad_performance) as f:
        list = [line.split(';') for line in f]
        list.pop(0)
        for elet in list:
            save_model_to = os.path.join(cfg.path_to_model, elet[1] + ".h5")
            list_model_name.append((elet[1],
                                    save_model_to))
    return list_model_name
