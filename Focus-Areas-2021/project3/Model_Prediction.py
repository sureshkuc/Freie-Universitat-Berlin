from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
%matplotlib inline

#trained model path
model = keras.models.load_model('model-best.h5')
model.load_weights('model-best.h5')
#images path that need to be tested
test_path =  '/home/suresh/Profile-Areas/Project3/DataSet/base_dir/test_dir'

# get the metric names so we can use evaulate_generator
model.metrics_names




def data_gen(test_path,IMAGE_SIZE_W = 300,IMAGE_SIZE_H=300,IMAGE_CHANNELS = 3):
    #IMAGE_SIZE_W = 300
    #IMAGE_SIZE_H=300
    #train_path = '/home/suresh/Profile-Areas/Project3/DataSet/base_dir/train_dir'
    #valid_path = '/home/suresh/Profile-Areas/Project3/DataSet/base_dir/val_dir'
    #test_path =  '/home/suresh/Profile-Areas/Project3/DataSet/base_dir/test_dir'

    

  
    datagen = ImageDataGenerator(rescale=1.0/255, #zoom_range=0.2,  # set range for random zoom
            rotation_range = 90,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True, )  # randomly flip images)

    
    # Note: shuffle=False causes the test dataset to not be shuffled
    test_gen = datagen.flow_from_directory(valid_path,
                                            target_size=(IMAGE_SIZE_W,IMAGE_SIZE_H),
                                            batch_size=1,
                                            class_mode='categorical',
                                            shuffle=False)
    return test_gen

test_gen=data_gen(test_path,IMAGE_SIZE_W = 150,IMAGE_SIZE_H=150,IMAGE_CHANNELS = 3)

val_loss, val_acc = model.evaluate(test_gen, steps=273)

print('val_loss:', val_loss)
print('val_acc:', val_acc)


predictions = model.predict_generator(test_gen, steps=273, verbose=1)

predictions

# This is how to check what index keras has internally assigned to each class. 
test_gen.class_indices

df_preds = pd.DataFrame(predictions, columns=['B', 'M'])

df_preds.head()

 #Get the true labels
y_true = test_gen.classes
#print(y_true)
# Get the predicted labels as probabilities
y_pred = df_preds['M']
#print(y_pred)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, predictions.argmax(axis=1))

cm_plot_labels = ['Benign', 'Malign']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

plt.hist(y_pred, bins = 10)
plt.title('CNN model Probabilities Distribution')
plt.show()

from sklearn.metrics import roc_curve, auc
fpr,tpr,thr=roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic Curve')
plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

from sklearn.metrics import classification_report

# Generate a classification report

# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)

