import os
import logging

# Set up logging configuration
LOG_DIR = "outputs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "application.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Dataset paths
TRAIN_PATH = '/home/suresh/Profile-Areas/Project3/DataSet/400X/'
TEST_PATH = '/home/suresh/Profile-Areas/Project3/DataSet/test/'
BASE_DIR = '/home/suresh/Profile-Areas/Project3/DataSet/base_dir'

# Image parameters
IMAGE_SIZE_W = 150
IMAGE_SIZE_H = 150
IMAGE_CHANNELS = 3

# Model configurations
MODEL_FILE_PATH = "model.h5"
CHECKPOINT_FILE_PATH = "Latest1_weights.best.hdf5"

