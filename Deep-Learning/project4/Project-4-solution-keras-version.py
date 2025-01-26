import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

with np.load('/home/suresh/SoSe2021/Deep-Learning/sose21/Assignments/Assignment4/prediction-challenge-02-data.npz') as fh:
    x_train = fh['x_train']
    y_train = fh['y_train']
    x_test = fh['x_test']
    

# TRAINING DATA: INPUT (x) AND OUTPUT (y)
# 1. INDEX: IMAGE SERIAL NUMBER (6000)
# 2/3. INDEX: PIXEL VALUE (32 x 32)
# 4. INDEX: COLOR CHANNELS (3)
print(x_train.shape, x_train.dtype)
print(y_train.shape, y_train.dtype)

# TEST DATA: INPUT (x) ONLY
print(x_test.shape, x_test.dtype)

X_train = x_train/ 255 
X_val = x_test / 255 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.30,random_state=42)

cifar10_model=tf.keras.models.Sequential()

# First Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same", activation="relu", input_shape=[32,32,3]))

# Second Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same", activation="relu"))

# Max Pooling Layer
cifar10_model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))

# Third Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))

# Fourth Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))

# Max Pooling Layer
cifar10_model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))

# Flattening Layer
cifar10_model.add(tf.keras.layers.Flatten())

# Droput Layer
cifar10_model.add(tf.keras.layers.Dropout(0.5,noise_shape=None,seed=None))

# Adding the first fully connected layer
cifar10_model.add(tf.keras.layers.Dense(units=128,activation='relu'))

# Output Layer
cifar10_model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

cifar10_model.summary()

cifar10_model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

history = cifar10_model.fit(X_train,y_train,epochs=10,validation_data=(X_test, y_test))

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

test_loss, test_accuracy = cifar10_model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

y_pred = cifar10_model.predict(X_test)


predictionmodel1 = [np.argmax(element) for element in y_pred]

from sklearn.metrics import confusion_matrix , classification_report
print("Classification Report: \n", classification_report(y_test, predictionmodel1))

y_pred2 = cifar10_model.predict(X_val)

predictionmodel2 = [np.argmax(element) for element in y_pred2]

prediction = np.array(predictionmodel2)

prediction.shape[0]

assert prediction.ndim == 1
assert prediction.shape[0] == 300

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('prediction.npy', prediction.astype(int))

def validate_prediction_format():
    loaded = np.load('prediction.npy')
    assert loaded.shape == (300, )
    assert loaded.dtype == int
    assert (loaded <= 2).all()
    assert (loaded >= 0).all()
validate_prediction_format()