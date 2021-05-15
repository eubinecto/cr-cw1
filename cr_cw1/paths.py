from os import path


HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_cr_cw1")
CIFAR10_DIR = path.join(PROJECT_DATA_DIR, 'cifar10')
MODELS_DIR = path.join(PROJECT_DATA_DIR, "models")  # this is where all the trained models are stored.


# MODEL_DIR files
BASE_CNN = ...
