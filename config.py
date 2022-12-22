# Data Generation
DATA_DIR = 'data'
DALLE_PATH = '' # Insert path to DALL-E pretrained model
RN_PATH = '' # Insert path to RN pretrained model
TRAIN_DATA_SIZE = 10000
VAL_DATA_SIZE = 1000
TEST_DATA_SIZE = 1000
DALLE_TRAIN_DATA_SIZE = 10000
IMAGE_SIZE = 64
OBJECT_SIZE = 5

# Data Loader
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
# model saving
MODEL_DIR = 'data/model'

# RN train configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 40,
    'lr': 1e-4,
}


# VAE train configuration
EPOCHS = 20
BATCH_SIZE_VAE = 8
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.98

NUM_TOKENS = 100
NUM_LAYERS = 1
NUM_RESNET_BLOCKS = 1
SMOOTH_L1_LOSS = False
EMB_DIM = 64
HID_DIM = 32
KL_LOSS_WEIGHT = 0

STARTING_TEMP = 1.
TEMP_MIN = 0.5
ANNEAL_RATE = 1e-6

NUM_IMAGES_SAVE = 4

# DALLE train configuration
VAE_PATH = './vae.pt'
DALLE_PATH = './dalle.pt'
TAMING = False    # use VAE from taming transformers paper
IMAGE_TEXT_FOLDER = './data/dalle/images'
BPE_PATH = None

EPOCHS = 20
BATCH_SIZE_DALLE = 4
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 1024
TEXT_SEQ_LEN = 256
DEPTH = 12
HEADS = 16
DIM_HEAD = 64

WANDB_KEY = None