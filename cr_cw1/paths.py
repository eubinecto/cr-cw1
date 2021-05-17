from os import path


HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_cr_cw1")
CIFAR10_DIR = path.join(PROJECT_DATA_DIR, 'cifar10')
MODELS_DIR = path.join(PROJECT_DATA_DIR, "models")  # this is where all the trained models are stored.
# the check point directories
BASE_CNN_DIR = path.join(MODELS_DIR, 'base_cnn')  # the state_dict of the base model.
TWO_CNN_DIR = path.join(MODELS_DIR, 'two_cnn')  # the state_dict of cnn with two conv layers.
THREE_CNN_DIR = path.join(MODELS_DIR, 'three_cnn')
