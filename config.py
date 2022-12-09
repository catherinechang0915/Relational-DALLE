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
CLIP_MODEL = 'ViT-B/32'

# Data Loader
TRAIN_DIR = '/mnt/data/data/train'
VAL_DIR = '/mnt/data/data/val'
TEST_DIR = '/mnt/data/data/test'
# model saving
MODEL_DIR = '/mnt/data/model'

# train configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 40,
    'lr': 1e-4,
    'relation_type': 'ternary'
}

WANDB_KEY = None