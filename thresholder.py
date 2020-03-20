import os
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics as skmetrics

from run import AVAILABLE_MODELS, MODEL_MAPPING, INPUT_SIZE, LOSS
from data_generator import load_data_memory


def test_thresholds():
    X, Y = load_data_memory(['data/test'], 'image', 'masks')
    
    allpixels = X.shape[0] * X.shape[1] * X.shape[2]
    thresholds = np.arange(0.3, 0.82, 0.02)
    results = []

    while True:
        m = input('Enter model name: ')
        if m == 'x':
            break

        graph_name = input('Enter name displayed in graph [{}]: '.format(m))
        if not graph_name:
            graph_name = m

        if m not in AVAILABLE_MODELS:
            print('Unknown model')
            continue

        w = input('Path to weights: ')
        if not os.path.isfile(w):
            print('Path to weights does not exist.')
            continue

        model = MODEL_MAPPING[m](input_size=INPUT_SIZE, loss=LOSS)

        try:
            model.load_weights(w)
        except ValueError:
            print('Failed to load weights')
            continue

        y_pred = []

        # make predictions
        for i in range(X.shape[0]):
            y_pred.append(model.predict(np.expand_dims(X[i], axis=0)))

        # ROC
        fpr, tpr, threshs = skmetrics.roc_curve(Y.flatten().astype('uint8'),
                                                np.array(y_pred).flatten())
        auc = skmetrics.auc(fpr, tpr)

        accs = []
        ious = []
        for thresh in thresholds:
            iou = 0
            okpixels = 0
            for i, _yp in enumerate(y_pred):
                yt = Y[i]
                yp = np.copy(_yp)

                yp[yp < thresh] = 0.0
                yp[yp >= thresh] = 1.0

                inter = np.sum(yt * yp)
                union_iou = np.sum(yt + yp) - inter
                iou += inter / union_iou

                okpixels += np.sum(yt == yp)
            
            ious.append((iou / X.shape[0]) * 100)
            accs.append((okpixels / allpixels) * 100)

        results.append({'name': graph_name, 'acc': accs, 'iou': ious,
                        'auc': auc, 'fpr': fpr, 'tpr': tpr})

    plt.figure()
    for i in results:
        plt.plot(thresholds, i['iou'], label=i['name'])
    plt.legend(loc='best')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.savefig('plots_other/thresholds_iou.png')

    plt.figure()
    for i in results:
        plt.plot(thresholds, i['acc'], label=i['name'])
    plt.legend(loc='best')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.savefig('plots_other/thresholds_acc.png')

    plt.figure()
    for i in results:
        plt.plot(i['fpr'], i['tpr'],
                 label='%s (AUC = %0.2f)' % (i['name'], i['auc']))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.savefig('plots_other/roc.png')


if __name__ == '__main__':
    test_thresholds()
