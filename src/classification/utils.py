import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_fraud_data():
    df = pd.read_csv('../../data/train.csv', sep='|')

    y = df.pop('fraud')
    df.insert(1, 'totalItemsScanned', df['scannedLineItemsPerSecond'] * df['totalScanTimeInSeconds'])
    X = df.drop(columns=[
        'scannedLineItemsPerSecond', 'lineItemVoidsPerPosition', 'valuePerSecond',
        'quantityModifications', 'grandTotal'
    ])

    return X, y


def score_evaluation(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    score = 0
    score += conf[0][1] * -25
    score += conf[1][0] * -5
    score += conf[1][1] * 5

    return score


def score_evaluation_norm(min_score, max_score):
    def score_evaluation_fn(y_true, y_pred):
        conf = confusion_matrix(y_true, y_pred)
        score = 0
        score += conf[0][1] * -25
        score += conf[1][0] * -5
        score += conf[1][1] * 5

        score = (score - min_score) / (max_score - min_score)
        return score

    return score_evaluation_fn


def find_best_thresh(predictor, X, y):
    thresholds, scores = find_threshold_scores(predictor, X, y)

    plt.xlabel('threshold')
    plt.ylabel('score')
    _ = plt.plot(thresholds, scores)
    max_score_index = np.argmax(scores)
    print('max score:', scores[max_score_index])
    print('threshold:', thresholds[max_score_index])

    max_score = scores[max_score_index]
    best_threshold = thresholds[max_score_index]

    return max_score, best_threshold


def find_threshold_scores(predictor, X, y):
    thresholds = []
    scores = []

    for thresh in range(1, 100, 5):
        thresh = thresh / 100

        thresh_scores = []

        # simulate many times
        for _ in range(1, 100):
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            y_pred = predictor(X_test)
            y_pred[y_pred >= thresh] = 1
            y_pred[y_pred < thresh] = 0

            thresh_scores.append(score_evaluation(y_test, y_pred))

        thresholds.append(thresh)
        score = np.mean(thresh_scores)
        scores.append(score)

    return thresholds, scores


def get_test_score(predictor, threshold):
    dftest = pd.read_csv('../../data/test.csv', sep='|')
    dftest.insert(1, 'totalItemsScanned', dftest['scannedLineItemsPerSecond'] * dftest['totalScanTimeInSeconds'])

    X_test = dftest.drop(columns=[
        'scannedLineItemsPerSecond', 'lineItemVoidsPerPosition', 'valuePerSecond',
        'quantityModifications', 'grandTotal'
    ])

    y_test = pd.read_csv('../../data/DMC-2019-realclass.csv', squeeze=True).to_numpy()

    y_pred = predictor(X_test)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    y_pred = y_pred.astype('int8')

    score = score_evaluation(y_test, y_pred)
    return score
