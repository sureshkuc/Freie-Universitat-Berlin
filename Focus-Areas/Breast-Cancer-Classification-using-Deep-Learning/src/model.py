import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import logging
from config import IMAGE_SIZE_W, IMAGE_SIZE_H, IMAGE_CHANNELS

# Proposed model
def build_proposed_model():
    try:
        kernel_size = (3,3)
        pool_size = (2,2)
        first_filters = 32
        second_filters = 64
        third_filters = 128
        dropout_conv = 0.3
        dropout_dense = 0.3

        model = Sequential()
        model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape=(IMAGE_SIZE_H, IMAGE_SIZE_W, IMAGE_CHANNELS)))
        model.add(Conv2D(first_filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(second_filters, kernel_size, activation='relu'))
        model.add(Conv2D(second_filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(third_filters, kernel_size, activation='relu'))
        model.add(Conv2D(third_filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_dense))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    except Exception as e:
        logging.error("Error building the proposed model: %s", e)
        raise

# Function for model callbacks
def get_callbacks():
    try:
        checkpoint = ModelCheckpoint(MODEL_FILE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00001)
        return [checkpoint, reduce_lr]
    except Exception as e:
        logging.error("Error getting callbacks: %s", e)
        raise

