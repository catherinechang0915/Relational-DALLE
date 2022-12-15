# Data Generation
TRAIN_DATA_GEN_DIR = 'data/train'
VAL_DATA_GEN_DIR = 'data/val'
TEST_DATA_GEN_DIR = 'data/test'
DALLE_PATH = '' # Insert path to DALL-E pretrained model
RN_PATH = '' # Insert path to RN pretrained model
TRAIN_DATA_SIZE = 10000
VAL_DATA_SIZE = 1000
TEST_DATA_SIZE = 1000
IMAGE_SIZE = 64
OBJECT_SIZE = 5

# Data Loader
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
# model saving
MODEL_DIR = 'data/model'

# train configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 40,
    'lr': 1e-4,
}

WANDB_KEY = None