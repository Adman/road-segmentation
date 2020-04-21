import random
import cv2
import numpy as np

from run import RESIZE_TO
from data_generator import normalize_image


def score_entropy_avg(inp):
    return np.average(inp)


def score_entropy_sum(inp):
    return np.sum(inp)


# SAMPLING STRATEGIES ===============================================
def strategy_random(images, k=1, **kwargs):
    return random.sample(images, k)


def _strategy_entropy(images, k=1, models=None, aggreg_func=None, **kwargs):
    scores = []
    active_models = list(filter(lambda x: not x.finished, models))
    for i in images:
        img = cv2.imread(i).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, RESIZE_TO)
        img = normalize_image(img, colorspace='rgb')

        # get average score
        score = 0
        for m in active_models:
            pred = m.model.predict(np.expand_dims(img, axis=0))[0]
            s = -(pred*np.log2(pred, where=0<pred) +
                  (1-pred)*np.log2(1-pred, where=0<(1-pred)))
            score += aggreg_func(s)
        score /= len(active_models)

        scores.append((i, score))

    return [i[0] for i in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]


def strategy_avg_entropy(images, k=1, models=None, **kwargs):
    return _strategy__entropy(images, k=k, models=models,
                              aggreg_func=score_entropy_avg, **kwargs)


def strategy_sum_entropy(images, k=1, models=None, **kwargs):
    return _strategy_entropy(images, k=k, models=models,
                             aggreg_func=score_entropy_sum, **kwargs)


# STOPPING STRATEGIES ===============================================
def stopping_nostop(model=None, **kwargs):
    return False


def stopping_early_val_loss(model=None, epochs=20, **kwargs):
    """Stop training if val loss has not decreased for some time"""
    best_ep = model.best_val_loss.epoch
    return best_ep < model.epochs_trained - epochs


# EPOCH STRATEGIES ==================================================
def epochs_constant(current, max_epochs=0, **kwargs):
    return current <= max_epochs

