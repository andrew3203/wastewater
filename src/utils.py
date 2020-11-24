import numpy as np


def get_best_threshold(y_true, scores, metric="precision"):
    """
    Parameters
    ----------
    y_true : numpy.ndarray
        1d vector, true target values.
    scores : numpy.ndarray
        1d vector, estimator`s probability predictions, scores
    metric : str
        'precision': choose threshold based on precision
        'recall': choose threshold based on recall
        'f1_score': choose threshold based on harmonic average of
                    precision and recall
        'sensitivity': choose threshold based on sensitivity
        'specificity': choose threshold based on specificity
        'med_f1_score': choose threshold based on harmonic average of
                    sensitivity and specificity
    

    Returns
    -------
    trials : dict
        'threshold' : list of used scores as thresholds
        'metrics' : list of metrics calculated with each
                    threshold in trials['threshold']
        'best_trial' : best threshold to use and it`s metric 
        
    
    Example
    -------
    >>> from src.utils import get_best_threshold

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_classification(n_samples=int(1e5))
    >>> clf = LogisticRegression().fit(X,y)
    >>> scores = clf.predict_proba(X)[:,1]
    array([0.95702791, 0.27423805, 0.9905933 , ..., 0.02838837, 0.0339324 , 0.97921772])

    >>> get_best_threshold(y, scores, metric="f1_score")["best_trial"] 
    {'threshold': 0.4613591717894563, 'metric': 0.9146460033587697}
    """
    y_true, scores = map(np.asarray, [y_true, scores])

    indices = scores.argsort()
    y_true, scores = y_true[indices], scores[indices]

    tp = (y_true == 1)[::-1].cumsum()[::-1]
    fp = (y_true == 0)[::-1].cumsum()[::-1]
    tn = (y_true == 0).cumsum()
    fn = (y_true == 1).cumsum()

    tp, fp, tn, fn = tp[1:], fp[1:], tn[:-1], fn[:-1]

    def precision(tp, fp, tn, fn):
        return tp / (tp + fp)

    def recall(tp, fp, tn, fn):
        return tp / (tp + fn)

    def f1_score(tp, fp, tn, fn):
        return 2 / (1 / precision(tp, fp, tn, fn) + 1 / recall(tp, fp, tn, fn))

    def sensitivity(tp, fp, tn, fn):
        return tp / (tp + fn)

    def specificity(tp, fp, tn, fn):
        return tn / (tn + fp)

    def med_f1_score(tp, fp, tn, fn):
        return 2 / (1 / sensitivity(tp, fp, tn, fn) + 1 / specificity(tp, fp, tn, fn))

    metric_func = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "med_f1_score": med_f1_score,
    }[metric]

    metric_vals = metric_func(tp, fp, tn, fn)
    best = metric_vals.argmax()

    trials = {
        "thresholds": scores[:-1],
        "metrics": metric_vals,
        "best_trial": {
            "threshold": scores[best],
            "metric": metric_vals[best],
        },
    }

    return trials
