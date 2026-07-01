"""Loads BCI Competition IV Dataset 2a (via moabb) and extracts
right-hand vs. left-hand motor-imagery epochs for a single subject.

This dataset is the public benchmark FBCSP was built and validated on
(Ang et al., 2008/2012), used here as a stand-in for the private stroke
patient EEG data in Cheng et al. 2020 (TBME) since that data isn't public.
"""
from __future__ import annotations

import mne
import numpy as np
from moabb.datasets import BNCI2014_001

TMIN, TMAX = 0.5, 2.5  # seconds relative to cue onset (per Ang et al. FBCSP paper)
EVENT_ID = {"left_hand": 1, "right_hand": 2}


def load_subject_epochs(subject: int, session: str = "0train") -> tuple[np.ndarray, np.ndarray, float]:
    """Return (X, y, sfreq) for one subject/session.

    X: (n_trials, n_channels, n_times) EEG epochs, left/right hand MI only.
    y: (n_trials,) labels, 0=left_hand, 1=right_hand.
    """
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject])
    session_data = data[subject][session]

    raws = list(session_data.values())
    raw = mne.concatenate_raws([r.copy() for r in raws])
    raw = raw.pick(picks="eeg")

    events, event_id = mne.events_from_annotations(raw)
    keep_id = {k: v for k, v in event_id.items() if k in EVENT_ID}
    epochs = mne.Epochs(
        raw, events, event_id=keep_id, tmin=TMIN, tmax=TMAX,
        baseline=None, preload=True, verbose=False,
    )

    X = epochs.get_data(copy=True)
    labels = epochs.events[:, 2]
    inv_map = {v: k for k, v in keep_id.items()}
    y = np.array([0 if inv_map[label] == "left_hand" else 1 for label in labels])
    return X, y, float(raw.info["sfreq"])


def subject_list() -> list[int]:
    return BNCI2014_001().subject_list


def load_subject_continuous(subject: int, session: str = "1test") -> tuple["mne.io.Raw", np.ndarray, dict]:
    """Return (raw, events, event_id) for one subject/session, unepoched.

    Used for pseudo-online simulation, where classification must run on a
    sliding window over continuous EEG rather than pre-cut trials.
    """
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject])
    session_data = data[subject][session]

    raws = list(session_data.values())
    raw = mne.concatenate_raws([r.copy() for r in raws])
    raw = raw.pick(picks="eeg")

    events, event_id = mne.events_from_annotations(raw)
    keep_id = {k: v for k, v in event_id.items() if k in EVENT_ID}
    keep_codes = set(keep_id.values())
    events = events[np.isin(events[:, 2], list(keep_codes))]
    return raw, events, keep_id


def load_rest_windows(subject: int, session: str = "0train", n_samples: int = 501) -> np.ndarray:
    """Extracts one pre-cue "rest" window per trial from a session's continuous
    EEG: `n_samples` immediately before each motor-imagery cue, when the
    subject is not yet performing motor imagery. Used to train a rest-vs-MI
    gate, since the offline trial epochs alone contain no negative/idle class.

    `n_samples` should match the MI epoch length from `load_subject_epochs`
    (501 samples = TMAX-TMIN inclusive at 250 Hz) so CSP sees equal-length
    trials for both classes.
    """
    raw, events, _ = load_subject_continuous(subject, session=session)
    data = raw.get_data()

    windows = []
    for sample, _, _ in events:
        start = sample - n_samples
        if start >= 0:
            windows.append(data[:, start:sample])
    return np.stack(windows, axis=0)
