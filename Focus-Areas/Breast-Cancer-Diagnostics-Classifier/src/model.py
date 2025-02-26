# model.py
class Model:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train_random_forest(self, n_estimators=10):
        try:
            clf_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=Config.RANDOM_STATE)
            clf_rf.fit(self.x_train, self.y_train)
            return clf_rf
        except Exception as e:
            logging.error(f"Error training RandomForest: {e}")

    def train_decision_tree(self):
        try:
            clf_tree = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
            clf_tree.fit(self.x_train, self.y_train)
            return clf_tree
        except Exception as e:
            logging.error(f"Error training DecisionTree: {e}")

    def train_xgboost(self):
        try:
            clf_xgb = xgb.XGBClassifier(objective='reg:logistic')
            clf_xgb.fit(self.x_train, self.y_train)
            return clf_xgb
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")

    def train_logistic_regression(self):
        try:
            clf_lr = LogisticRegression(random_state=Config.RANDOM_STATE)
            clf_lr.fit(self.x_train, self.y_train)
            return clf_lr
        except Exception as e:
            logging.error(f"Error training LogisticRegression: {e}")


