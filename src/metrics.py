from sklearn.metrics import roc_auc_score, average_precision_score


def roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)
