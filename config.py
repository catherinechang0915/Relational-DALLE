# Data Generation
TRAIN_DATA_GEN_DIR = 'data/train'
VAL_DATA_GEN_DIR = 'data/val'
TEST_DATA_GEN_DIR = 'data/test'
TRAIN_DATA_SIZE = 20
VAL_DATA_SIZE = 10
TEST_DATA_SIZE = 10
IMAGE_SIZE = 256
OBJECT_SIZE = 12
CLIP_MODEL = 'ViT-B/32'

# Data Loader
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

# train configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 1e-3,
    'relation_type': 'binary'
}