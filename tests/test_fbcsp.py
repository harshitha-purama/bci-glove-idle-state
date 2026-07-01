import numpy as np
import pytest

from bci_glove.fbcsp import CSP, FBCSP
from bci_glove.preprocessing import FilterBank


def make_separable_classes(n_trials=60, n_channels=4, n_times=500, seed=0):
    """Two classes whose difference lies purely in the variance of channel 0
    (the classic case CSP is designed to separate)."""
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_trials, n_channels, n_times))
    X1 = rng.standard_normal((n_trials, n_channels, n_times))
    X0[:, 0, :] *= 3.0  # class 0 has high variance on channel 0
    X1[:, 0, :] *= 0.3  # class 1 has low variance on channel 0
    y = np.concatenate([np.zeros(n_trials), np.ones(n_trials)]).astype(int)
    X = np.concatenate([X0, X1], axis=0)
    return X, y


def test_csp_separates_variance_difference():
    X, y = make_separable_classes()
    csp = CSP(n_components=2).fit(X[y == 0], X[y == 1])
    feats = csp.transform(X)
    # first log-variance feature should differ strongly by class
    assert feats[y == 0, 0].mean() > feats[y == 1, 0].mean()


def test_csp_filters_shape():
    X, y = make_separable_classes(n_channels=6)
    csp = CSP(n_components=2).fit(X[y == 0], X[y == 1])
    assert csp.filters_.shape == (4, 6)  # 2*n_components filters, n_channels wide


def test_filterbank_output_shape():
    X, _ = make_separable_classes(n_times=500)
    fb = FilterBank(sfreq=250.0)
    out = fb.transform(X)
    assert out.shape == (len(fb.bands), *X.shape)


def test_fbcsp_end_to_end_beats_chance():
    X, y = make_separable_classes(n_trials=40, n_times=500)
    fbcsp = FBCSP(sfreq=250.0, n_csp_components=2, n_features_select=4)
    feats_train = fbcsp.fit_transform(X, y)
    assert feats_train.shape == (80, 4)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB().fit(feats_train, y)
    acc = clf.score(feats_train, y)
    assert acc > 0.9  # near-perfectly separable synthetic signal


def test_fbcsp_selected_features_within_bounds():
    X, y = make_separable_classes(n_trials=30, n_channels=4, n_times=500)
    fbcsp = FBCSP(sfreq=250.0, n_csp_components=2, n_features_select=4)
    fbcsp.fit(X, y)
    n_bands = len(fbcsp.bands)
    n_candidate_feats = n_bands * 2 * fbcsp.n_csp_components
    assert fbcsp.selected_idx_.max() < n_candidate_feats
    assert len(fbcsp.selected_idx_) == 4
