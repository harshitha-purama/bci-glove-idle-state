import numpy as np

from bci_glove.fbcsp import FBCSP
from bci_glove.online_sim import CONFIDENCE_THRESHOLD, _slide_windows, simulate_subject


class FakeRaw:
    """Minimal stand-in for mne.io.Raw exposing only what online_sim needs."""

    def __init__(self, data: np.ndarray, sfreq: float):
        self._data = data
        self.info = {"sfreq": sfreq}

    def get_data(self):
        return self._data


def test_slide_windows_shapes():
    sfreq = 250.0
    data = np.random.default_rng(0).standard_normal((4, int(4 * sfreq)))
    windows, times = _slide_windows(data, sfreq, t_start=0.0, t_end=4.0, window_sec=2.0, step_sec=0.5)
    assert windows.shape[1:] == (4, 500)
    assert len(times) == windows.shape[0]
    assert np.allclose(np.diff(times), 0.5)


def _make_continuous_signal(sfreq=250.0, n_channels=4, total_sec=60.0, cue_samples=(2500, 7500, 12500), seed=0):
    """Builds a synthetic continuous recording where, starting exactly at each
    cue sample, channel 0's variance jumps up for label 0 cues and channel 1's
    variance jumps up for label 1 cues, alternating, so a CSP-based decoder
    should detect it almost immediately with near-zero false triggers.
    """
    rng = np.random.default_rng(seed)
    n_times = int(total_sec * sfreq)
    data = rng.standard_normal((n_channels, n_times)) * 0.5
    labels = []
    for i, cue in enumerate(cue_samples):
        label = i % 2
        labels.append(label)
        end = min(n_times, cue + int(4 * sfreq))
        data[0 if label == 0 else 1, cue:end] *= 6.0
    return data, labels


def _fit_fbcsp_and_classifier():
    from sklearn.naive_bayes import GaussianNB

    fbcsp = FBCSP(sfreq=250.0, n_csp_components=1, n_features_select=2)
    n_trials = 20
    rng = np.random.default_rng(1)
    X0 = rng.standard_normal((n_trials, 4, 500)) * 0.5
    X0[:, 0, :] *= 6.0
    X1 = rng.standard_normal((n_trials, 4, 500)) * 0.5
    X1[:, 1, :] *= 6.0
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n_trials), np.ones(n_trials)]).astype(int)
    feats = fbcsp.fit_transform(X, y)
    clf = GaussianNB().fit(feats, y)
    return fbcsp, clf


def test_simulate_subject_returns_one_result_per_trial_in_order():
    sfreq = 250.0
    data, labels = _make_continuous_signal(sfreq=sfreq)
    raw = FakeRaw(data, sfreq)
    events = np.array([[2500, 0, 10], [7500, 0, 11], [12500, 0, 10]])
    event_id = {"a": 10, "b": 11}
    label_map = {10: 0, 11: 1}

    fbcsp, clf = _fit_fbcsp_and_classifier()
    summary = simulate_subject(fbcsp, clf, raw, events, event_id, label_map)

    assert len(summary.trials) == 3
    assert [t.label for t in summary.trials] == labels
    assert 0.0 <= summary.detection_rate <= 1.0
    assert 0.0 <= summary.false_trigger_rate_per_trial <= 1.0
    # With strongly separable synthetic signal, the decoder should catch at least one trial.
    assert summary.detection_rate > 0


def test_zero_confidence_threshold_forces_every_trial_to_false_trigger(monkeypatch):
    import bci_glove.online_sim as online_sim_module

    monkeypatch.setattr(online_sim_module, "CONFIDENCE_THRESHOLD", 0.0)

    sfreq = 250.0
    data, labels = _make_continuous_signal(sfreq=sfreq)
    raw = FakeRaw(data, sfreq)
    events = np.array([[2500, 0, 10], [7500, 0, 11], [12500, 0, 10]])
    event_id = {"a": 10, "b": 11}
    label_map = {10: 0, 11: 1}

    fbcsp, clf = _fit_fbcsp_and_classifier()
    summary = online_sim_module.simulate_subject(fbcsp, clf, raw, events, event_id, label_map)

    # threshold 0 means every window's max-probability trivially clears the bar
    assert summary.false_trigger_rate_per_trial == 1.0
