def get_best_threshold(y_true, scores, metric="precision"):
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
