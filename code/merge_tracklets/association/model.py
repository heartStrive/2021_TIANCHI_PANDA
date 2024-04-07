from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from joblib import dump, load
from sklearn.metrics import classification_report


class TruePositiveClassifier:
    def __init__(self, model_path = None):
        if model_path is None:
            self.clf = GaussianNB()
        else:
            self.clf = load(model_path)
    
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def infer(self, X, proba = True):
        if proba:
            y_pred = self.clf.predict_proba(X)[:,1]
        else:
            y_pred = self.clf.predict(X)
        return y_pred
    
    def eval(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))

    def save_model(self, out_path):
        dump(self.clf, out_path)

class StatusClassifier:
    def __init__(self, model_path = None):
        if model_path is None:
            self.clf = MLPClassifier(random_state=1, hidden_layer_sizes=20, max_iter=2000)
        else:
            self.clf = load(model_path)
    
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def infer(self, X, proba=True):
        if proba:
            y_pred = self.clf.predict_proba(X)
        else:
            y_pred = self.clf.predict(X)
        return y_pred
    
    def eval(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))
    
    def save_model(self, out_path):
        dump(self.clf, out_path)