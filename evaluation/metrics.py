from sklearn import metrics


def f1_score(target, predicted, average):
    """
    Args:
        target: ground truth
        predicted: preictions from the model
        average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
    Returns:
        A list of terms and the respective tfidf weight (zero values will be ignored)
    """
    return metrics.f1_score(target, predicted, average=average)


def accuracy(target, predicted):
     return metrics.accuracy_score(target, predicted)


def precision(target, predicted, average):
     return metrics.precision_score(target, predicted, average=average)


def recall(target, predicted, average):
     return metrics.recall_score(target, predicted, average=average)


