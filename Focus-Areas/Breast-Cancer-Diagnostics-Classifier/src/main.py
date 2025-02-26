from evaluation import Evaluation
from model import Model
from config import Config
# main.py
def main():
    try:
        # Load Data
        data = pd.read_csv(Config.DATA_PATH, names=Config.COLUMN_NAMES)
        data = data.drop(['id'], axis=1)
        y = data.diagnosis
        data = data.drop(['diagnosis'], axis=1)
        data = data.drop(Config.DROP_COLUMNS, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
        
        # Train Models
        model = Model(x_train, y_train)
        rf_model = model.train_random_forest()
        dt_model = model.train_decision_tree()
        xgb_model = model.train_xgboost()
        lr_model = model.train_logistic_regression()
        
        # Evaluate Models
        print("Random Forest Model:")
        Evaluation.evaluate_model(rf_model, x_test, y_test)
        
        print("Decision Tree Model:")
        Evaluation.evaluate_model(dt_model, x_test, y_test)
        
        print("XGBoost Model:")
        Evaluation.evaluate_model(xgb_model, x_test, y_test)
        
        print("Logistic Regression Model:")
        Evaluation.evaluate_model(lr_model, x_test, y_test)
        
        # Plot Confusion Matrix
        Evaluation.plot_confusion_matrix(rf_model, x_test, y_test)
        
        # Plot ROC Curve
        Evaluation.plot_roc_curve(rf_model, x_test, y_test)
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
