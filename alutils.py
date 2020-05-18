import random
import cv2
import numpy as np

from keras.models import Model
from sklearn.cluster import KMeans
from umap import UMAP

from run import RESIZE_TO
from data_generator import normalize_image


# HELPERS ===========================================================
def get_last_encoder_layer_by_model(modelname, name_mapping):
    return {
        'ResNet-4': 'activation_25',
        'MNetV3-S-2': 'final_activation',
        'SNetV2-1': 'stage4/block4/channel_shuffle'
    }[name_mapping[modelname]]


def score_entropy_avg(inp):
    return np.average(inp)


def score_entropy_sum(inp):
    return np.sum(inp)


def load_and_normalize(i):
    img = cv2.imread(i).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, RESIZE_TO)
    return normalize_image(img, colorspace='rgb')


def get_entropy_scores(active_models, images, aggreg_func):
    """Return array of tuples (image name, entropy score)"""
    scores = []
    for i in images:
        img = load_and_normalize(i)

        # get average score
        score = 0
        for m in active_models:
            pred = m.model.predict(np.expand_dims(img, axis=0))[0]
            s = -(pred*np.log2(pred, where=0<pred) +
                  (1-pred)*np.log2(1-pred, where=0<(1-pred)))
            score += aggreg_func(s)
        score /= len(active_models)

        scores.append((i, score))
    return scores


# SAMPLING STRATEGIES ===============================================
def strategy_random(images, k=1, **kwargs):
    return random.sample(images, k)


def _strategy_entropy(images, k=1, models=None, aggreg_func=None, **kwargs):
    scores = []
    active_models = list(filter(lambda x: not x.finished, models))

    scores = get_entropy_scores(active_models, images, aggreg_func)
    return [i[0] for i in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]


def strategy_avg_entropy(images, k=1, models=None, **kwargs):
    return _strategy_entropy(images, k=k, models=models,
                             aggreg_func=score_entropy_avg, **kwargs)


def strategy_sum_entropy(images, k=1, models=None, **kwargs):
    return _strategy_entropy(images, k=k, models=models,
                             aggreg_func=score_entropy_sum, **kwargs)


def strategy_diversity(images, k=1, models=None, name_mapping=None, **kwargs):
    # we expect that models and modelnames is just array of one item
    model = models[0]
    layer_name = get_last_encoder_layer_by_model(model.name, name_mapping)
    out = [model.model.get_layer(layer_name).output]
    _m = Model(inputs=model.model.inputs, outputs=out)

    scores = get_entropy_scores(models, images, score_entropy_avg)

    # compute embedding for every image
    embeddings = []
    for i in images:
        img = load_and_normalize(i)

        pred = _m.predict(np.expand_dims(img, axis=0))[0]
        avg = np.mean(pred, axis=2)
        flat = avg.flatten()
        embeddings.append(flat)

    embeddings = np.array(embeddings)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)

    pred_classes = kmeans.predict(embeddings)

    scores_np = np.array(scores, dtype=tuple)
    result = []
    for cluster in range(k):
        clust = list(scores_np[np.where(pred_classes == cluster)])
        if len(clust):
            result.append(
                sorted(clust, key=lambda x: x[1], reverse=True)[0][0]
            )

    return result


def strategy_init_diversity(images, k=1, **kwargs):
    vectors = []
    for i in images:
        img = load_and_normalize(i)
        vectors.append(img.flatten())

    vectors = np.array(vectors)
    # 640*480 vectors
    embeddings = UMAP(n_components=2).fit_transform(vectors)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)

    pred_classes = kmeans.predict(embeddings)
    images_np = np.array(images, dtype=str)
    result = []
    for cluster in range(k):
        clust = list(images_np[np.where(pred_classes == cluster)])
        if len(clust):
            result.append(clust[0])

    return result


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

