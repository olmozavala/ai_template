from keras.optimizers import *
from keras.metrics import *
from os.path import join
import os

from AI.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
_data_folder = '/media/osz1/DATA/DATA/PX/'  # Where the data is stored and where the preproc folder will be saved
_preproc_folder = 'Preproc'  # Name to save preprocessed data
_run_name = F'Prostate_MultiStream_{_preproc_folder}'  # Name of the model, for training and classification
_output_folder = '/media/osz1/DATA/DATA/DELETE/NEW'  # Where to save the models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.HALF_UNET_2D_SINGLE_STREAM_CLASSIFICATION,
        ModelParams.DROPOUT: True,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [28, 28],
        ModelParams.START_NUM_FILTERS: 32,
        ModelParams.NUMBER_LEVELS: 2,
        ModelParams.FILTER_SIZE: 3,
        ModelParams.NUMBER_DENSE_LAYERS: 2,
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 10
    }
    return {**cur_config, **model_config}


def get_training_2d():
    cur_config = {
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.evaluation_metrics: [sparse_categorical_accuracy],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: sparse_categorical_crossentropy,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.epochs: 10,
        TrainingParams.config_name: _run_name,
    }
    return append_model_params(cur_config)

