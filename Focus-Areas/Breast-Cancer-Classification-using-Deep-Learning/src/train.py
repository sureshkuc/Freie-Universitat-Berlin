import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_PATH, BASE_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H

# Data generation function
def data_gen():
    try:
        datagen = ImageDataGenerator(rescale=1.0/255)
        train_gen = datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=(IMAGE_SIZE_W, IMAGE_SIZE_H),
            batch_size=32,
            class_mode='categorical'
        )
        return train_gen
    except Exception as e:
        logging.error("Error during data generation: %s", e)
        raise

# Training the model
def train_model(model, train_gen):
    try:
        steps_per_epoch = np.ceil(len(train_gen) / 32)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            verbose=1
        )
        return history
    except Exception as e:
        logging.error("Error during training: %s", e)
        raise

