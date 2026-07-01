"""From-scratch implementation of Common Spatial Patterns (CSP) and
Filter Bank CSP (FBCSP), per:

Ang, K.K., Chin, Z.Y., Zhang, H., Guan, C. (2008). "Filter Bank Common
Spatial Pattern (FBCSP) in Brain-Computer Interface." IJCNN 2008.

Pipeline: bandpass filter bank -> per-band CSP -> log-variance features ->
mutual-information feature selection (MIBIF) -> classifier.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.feature_selection import mutual_info_classif

from .preprocessing import DEFAULT_BANDS, FilterBank


class CSP:
    """Binary-class Common Spatial Patterns.

    Spatial filters are the generalized eigenvectors of the two classes'
    averaged, trace-normalized spatial covariance matrices. The filters
    that maximize variance for class 0 simultaneously minimize it for
    class 1, and vice versa.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components  # filter pairs kept from each end of the spectrum
        self.filters_: np.ndarray | None = None  # (2*n_components, n_channels)

    @staticmethod
    def _mean_covariance(X: np.ndarray) -> np.ndarray:
        # X: (n_trials, n_channels, n_times) -> trace-normalized spatial covariance, averaged over trials
        covs = np.einsum("ncj,ndj->ncd", X, X)
        traces = np.trace(covs, axis1=1, axis2=2)
        covs = covs / traces[:, None, None]
        return covs.mean(axis=0)

    def fit(self, X0: np.ndarray, X1: np.ndarray) -> "CSP":
        C0 = self._mean_covariance(X0)
        C1 = self._mean_covariance(X1)
        # Generalized eigenvalue problem: C0 w = lambda (C0 + C1) w
        eigvals, eigvecs = eigh(C0, C0 + C1)
        order = np.argsort(eigvals)[::-1]  # descending: best-for-class-0 first
        eigvecs = eigvecs[:, order]
        n = self.n_components
        self.filters_ = np.concatenate([eigvecs[:, :n], eigvecs[:, -n:]], axis=1).T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_trials, n_channels, n_times) -> log-variance features (n_trials, 2*n_components)
        projected = np.einsum("fc,ncj->nfj", self.filters_, X)
        var = projected.var(axis=-1)
        var = var / var.sum(axis=-1, keepdims=True)
        return np.log(var)


class FBCSP:
    """Filter Bank CSP: CSP per sub-band, then mutual-information feature selection."""

    def __init__(
        self,
        sfreq: float,
        bands: list[tuple[float, float]] | None = None,
        n_csp_components: int = 2,
        n_features_select: int = 4,
        random_state: int = 0,
    ):
        self.sfreq = sfreq
        self.bands = bands or DEFAULT_BANDS
        self.n_csp_components = n_csp_components
        self.n_features_select = n_features_select
        self.random_state = random_state
        self.filter_bank_ = FilterBank(sfreq, self.bands)
        self.csp_per_band_: list[CSP] = []
        self.selected_idx_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSP":
        Xb = self.filter_bank_.transform(X)  # (n_bands, n_trials, n_channels, n_times)
        self.csp_per_band_ = []
        all_feats = []
        for b in range(Xb.shape[0]):
            csp = CSP(self.n_csp_components).fit(Xb[b, y == 0], Xb[b, y == 1])
            self.csp_per_band_.append(csp)
            all_feats.append(csp.transform(Xb[b]))
        feats = np.concatenate(all_feats, axis=1)  # (n_trials, n_bands * 2*n_components)

        mi = mutual_info_classif(feats, y, random_state=self.random_state)
        n_select = min(self.n_features_select, feats.shape[1])
        self.selected_idx_ = np.argsort(mi)[::-1][:n_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xb = self.filter_bank_.transform(X)
        all_feats = [csp.transform(Xb[b]) for b, csp in enumerate(self.csp_per_band_)]
        feats = np.concatenate(all_feats, axis=1)
        return feats[:, self.selected_idx_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
