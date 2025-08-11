import numpy as np
from src.model import train_logit, predict_proba


def test_train_predict_shapes():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    m = train_logit(X, y)
    scores = predict_proba(m, X)
    assert scores.shape == (4,)
