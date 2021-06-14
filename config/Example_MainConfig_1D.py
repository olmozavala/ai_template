from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from os.path import join
import os

from AI_proj.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
_data_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/data'  # Where the data is stored and where the preproc folder will be saved
_run_name = F'Current_Run'  # Name of the model, for training and classification
_output_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output'  # Where to save the models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: False,
        ModelParams.INPUT_SIZE: 4,
        ModelParams.HIDDEN_LAYERS: 3,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [8,8,8],
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 1,
    }
    return {**cur_config, **model_config}


def get_training_1d():
    cur_config = {
        TrainingParams.input_folder: _data_folder,
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.file_name: 'SWS2forML_nowave.csv',
        TrainingParams.evaluation_metrics: [mse],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 30,
        TrainingParams.epochs: 1000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False
    }
    return append_model_params(cur_config)


def get_usemodel_1d_config():
    cur_config = {
        ClassificationParams.input_folder: _data_folder,
        ClassificationParams.output_folder: F"{join(_output_folder, 'Results', _run_name)}",
        # ClassificationParams.model_weights_file: '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output/Training/Current_Run/models/Current_Run_2019_10_11_17_32-01-0.06.hdf5',
        ClassificationParams.model_weights_file: '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output/Training/Current_Run/models/Current_Run_2019_10_11_20_45-01-0.03.hdf5',
        ClassificationParams.output_file_name: 'Results.csv',
        ClassificationParams.input_file: 'SWS2forML_nowave.csv',
        ClassificationParams.output_imgs_folder: F"{join(_output_folder, 'Results', _run_name, 'imgs')}",
        ClassificationParams.show_imgs: True,
        ClassificationParams.metrics: [ClassificationMetrics.MSE],
    }
    return append_model_params(cur_config)
