"""Cross-validated evaluation of FBCSP + classifier, matching the
per-subject accuracy reporting style of Ang et al.'s FBCSP benchmarks.
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from .fbcsp import FBCSP

CLASSIFIERS = {
    "gnb": lambda: GaussianNB(),  # closest sklearn stand-in for the paper's Parzen-window Bayes classifier
    "lda": lambda: LinearDiscriminantAnalysis(),
    "svm": lambda: SVC(kernel="linear"),
}


def cross_validate_subject(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    classifier: str = "gnb",
    n_splits: int = 10,
    n_csp_components: int = 2,
    n_features_select: int = 4,
    random_state: int = 0,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        fbcsp = FBCSP(
            sfreq,
            n_csp_components=n_csp_components,
            n_features_select=n_features_select,
            random_state=random_state,
        )
        feats_train = fbcsp.fit_transform(X[train_idx], y[train_idx])
        feats_test = fbcsp.transform(X[test_idx])

        clf = CLASSIFIERS[classifier]()
        clf.fit(feats_train, y[train_idx])
        accs.append(clf.score(feats_test, y[test_idx]))

    accs = np.array(accs)
    return {"fold_accuracies": accs, "mean_accuracy": accs.mean(), "std_accuracy": accs.std()}
