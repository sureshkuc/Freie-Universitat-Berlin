# evaluation.py
class Evaluation:
    @staticmethod
    def evaluate_model(model, x_test, y_test):
        try:
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {acc}')
            print(classification_report(y_test, y_pred))
            return acc
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
    @staticmethod
    def plot_confusion_matrix(model, x_test, y_test):
        try:
            metrics.plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")

    @staticmethod
    def plot_roc_curve(model, x_test, y_test):
        try:
            metrics.plot_roc_curve(model, x_test, y_test)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {e}")


