import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logit(X, y):
    """Train a simple logistic regression with class_weight='balanced'."""
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    return clf.fit(X, y)


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]
