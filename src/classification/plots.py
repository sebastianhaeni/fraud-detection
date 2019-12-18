import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import plot_roc_curve, auc, plot_confusion_matrix, plot_precision_recall_curve, make_scorer, \
    confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve


def plot_classifier(clf, X, y):
    plot_fraud_confusion_matrix(clf, X, y)
    plot_precision_recall_curve(clf, X, y)
    plot_cv_roc_curve(clf, X, y)
    plot_learning_curve(clf, X, y, cv=StratifiedKFold(n_splits=5))


def plot_fraud_confusion_matrix(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = clf.fit(X_train, y_train)
    plot_confusion_matrix(model, X_test, y_test, normalize='all')


def plot_cv_roc_curve(clf, X, y):
    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X.iloc[train], y.iloc[train])
        viz = plot_roc_curve(clf,
                             X.iloc[test],
                             y.iloc[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3,
                             lw=1,
                             ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='random',
            alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color='grey',
                    alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05],
           ylim=[-0.05, 1.05],
           title="ROC")
    ax.legend(loc="lower right")
    plt.show()


def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    ax = plt.axes()

    ax.set_title('Learning curves')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       scoring=make_scorer(score_evaluation))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax.legend(loc="best")

    return plt


def score_evaluation(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    score = 0
    score += conf[0][1] * -25
    score += conf[1][0] * -5
    score += conf[1][1] * 5

    return score
