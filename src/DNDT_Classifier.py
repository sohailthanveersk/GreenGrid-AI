# ============================
# DNDT_Classifier.py
# ============================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os

class DNDTClassifier:
    def __init__(self, model_path,X_train,X_test):
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self.model = load(model_path)
        else:
            self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)
        dump(self.model, self.model_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
