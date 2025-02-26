import os
import logging
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
from sklearn.metrics import classification_report

# Setup logging
log_directory = './outputs'
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_directory, 'model_evaluation.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self, model_path, test_path, image_size=(300, 300)):
        self.model_path = model_path
        self.test_path = test_path
        self.image_size = image_size
        self.model = None
        self.test_gen = None
    
    def load_model(self):
        try:
            logging.info(f"Loading model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            self.model.load_weights(self.model_path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def data_gen(self):
        try:
            logging.info(f"Preparing test data generator for test path: {self.test_path}")
            datagen = ImageDataGenerator(rescale=1.0/255, 
                                         rotation_range=90, 
                                         horizontal_flip=True, 
                                         vertical_flip=True)
            self.test_gen = datagen.flow_from_directory(self.test_path,
                                                       target_size=self.image_size,
                                                       batch_size=1,
                                                       class_mode='categorical',
                                                       shuffle=False)
            logging.info("Test data generator created successfully")
        except Exception as e:
            logging.error(f"Error creating test data generator: {e}")
            raise
    
    def evaluate_model(self):
        try:
            logging.info("Evaluating model")
            val_loss, val_acc = self.model.evaluate(self.test_gen, steps=273)
            logging.info(f"Evaluation results - Loss: {val_loss}, Accuracy: {val_acc}")
            return val_loss, val_acc
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise
    
    def generate_predictions(self):
        try:
            logging.info("Generating predictions")
            predictions = self.model.predict(self.test_gen, steps=273, verbose=1)
            logging.info(f"Predictions generated successfully")
            return predictions
        except Exception as e:
            logging.error(f"Error generating predictions: {e}")
            raise

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        try:
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                logging.info("Normalized confusion matrix")
            else:
                logging.info("Confusion matrix, without normalization")
            logging.info(f"Confusion Matrix: \n{cm}")
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
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            raise

    def plot_roc_curve(self, y_true, y_pred):
        try:
            fpr, tpr, thr = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic Curve')
            plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc='lower right')
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {e}")
            raise

    def generate_classification_report(self, y_true, y_pred_binary):
        try:
            logging.info("Generating classification report")
            report = classification_report(y_true, y_pred_binary, target_names=['Benign', 'Malign'])
            logging.info(f"Classification Report: \n{report}")
            return report
        except Exception as e:
            logging.error(f"Error generating classification report: {e}")
            raise

def main():
    try:
        # Paths
        model_path = 'model-best.h5'
        test_path = '/home/suresh/Profile-Areas/Project3/DataSet/base_dir/test_dir'

        # Create ModelEvaluator instance
        evaluator = ModelEvaluator(model_path, test_path, image_size=(150, 150))

        # Load model
        evaluator.load_model()

        # Prepare test data generator
        evaluator.data_gen()

        # Evaluate model
        val_loss, val_acc = evaluator.evaluate_model()

        # Generate predictions
        predictions = evaluator.generate_predictions()

        # Get true labels
        y_true = evaluator.test_gen.classes

        # Get predicted labels as probabilities
        y_pred = pd.DataFrame(predictions, columns=['B', 'M'])['M']

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_true, y_pred)
        logging.info(f"ROC AUC Score: {roc_auc}")

        # Plot confusion matrix
        cm = confusion_matrix(y_true, predictions.argmax(axis=1))
        evaluator.plot_confusion_matrix(cm, classes=['Benign', 'Malign'])

        # Plot ROC curve
        evaluator.plot_roc_curve(y_true, y_pred)

        # Generate and log classification report
        y_pred_binary = predictions.argmax(axis=1)
        report = evaluator.generate_classification_report(y_true, y_pred_binary)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == '__main__':
    main()

