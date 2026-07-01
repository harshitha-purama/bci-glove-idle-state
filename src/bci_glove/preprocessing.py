"""Filter bank used by FBCSP: splits EEG into overlapping-free sub-bands.

Per Ang et al. (2008), "Filter Bank Common Spatial Pattern (FBCSP) in
Brain-Computer Interface": 9 bands of 4 Hz width spanning 4-40 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

DEFAULT_BANDS = [(f, f + 4) for f in range(4, 40, 4)]  # 4-8, 8-12, ..., 36-40


class FilterBank:
    def __init__(self, sfreq: float, bands: list[tuple[float, float]] | None = None, order: int = 4):
        self.sfreq = sfreq
        self.bands = bands or DEFAULT_BANDS
        self.order = order

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n_trials, n_channels, n_times) -> (n_bands, n_trials, n_channels, n_times)."""
        nyq = self.sfreq / 2.0
        out = np.empty((len(self.bands), *X.shape), dtype=X.dtype)
        for i, (lo, hi) in enumerate(self.bands):
            b, a = butter(self.order, [lo / nyq, hi / nyq], btype="bandpass")
            out[i] = filtfilt(b, a, X, axis=-1)
        return out
