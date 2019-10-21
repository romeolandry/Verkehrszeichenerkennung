import config as cfg

from models.Optimisation_TRt import Optimisation_TRT
from optparse import OptionParser


parser = OptionParser()

parser.add_option("-p", "--path", dest="pfad_keras_model",
                  help="pafd zum trainierten Kears-model",
                  default=cfg.path_h5_model)

parser.add_option("-d", "--dest",
                  dest="pfad_trt_model",
                  help="ordner von TensorRT-model",
                  default=cfg.path_rt_opt_model)

parser.add_option("-s", "--save",
                  dest="pfad_tf_model",
                  help="pfad zur Speicherung des Tf-Models",
                  default=cfg.path_tf_model)

parser.add_option("-m", "--mode",
                  dest="trainned_config",
                  help="yes, wenn das Model mit Bachnormalisation" +
                  "trainniert wurde",
                  default=False)

(options, args) = parser.parse_args()

pfad_keras_model = options.pfad_keras_model
pfad_trt_model = options.pfad_trt_model
pfad_tf_model = options.pfad_tf_model
trainned_config = options.trainned_config

rt_optimizer = Optimisation_TRT(pfad_keras_model,
                                pfad_tf_model,
                                pfad_trt_model,
                                cfg.path_to_frozen_model)
# Keras-Model umwandelt
rt_optimizer.keras_to_tensor_model(trainned_config)

# optimierung des Models
# 1- tf_model -> frozen_model.pb
rt_optimizer.conver_tf_to_frozen_model()
