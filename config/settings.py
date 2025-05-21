IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

EPOCHS = 30
LEARNING_RATE = 0.001

DATA_DIR = "/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables"

MODEL_SAVE_PATH = 'fruit_freshness_model.h5'
BEST_MODEL_PATH = 'best_model.h5'

AUGMENTATION_CONFIG = {
    'rescale': 1./255,
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'validation_split': 0.2
}