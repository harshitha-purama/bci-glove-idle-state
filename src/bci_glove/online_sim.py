"""Pseudo-online simulation of the FBCSP decoder.

The offline benchmark classifies pre-cut, hand-picked 2s windows. It never
answers the question that actually matters for driving a physical glove:
if you run the decoder continuously, how long after the cue does it commit
to the right answer, and how often does it fire on nothing (rest)?

This module slides a fixed-length window across continuous EEG around each
motor-imagery cue and applies the already-trained FBCSP + classifier at
each step, using a confidence threshold and a debounce rule (N consecutive
confident-and-agreeing windows) to decide when a "trigger" would fire.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fbcsp import FBCSP

WINDOW_SEC = 2.0        # matches training epoch length (TMIN..TMAX in data.py)
STEP_SEC = 0.2          # 5 Hz decision rate
PRE_CUE_SEC = 3.4        # rest span before each cue: 8 windows, so the debounce can apply.
                         # Reaches 1.4 s before the fixation cross; check it stays clear of
                         # the previous trial for this dataset before trusting the numbers.
POST_CUE_SEC = 6.0       # how long after the cue we allow for a detection
CONFIDENCE_THRESHOLD = 0.7
GATE_CONFIDENCE_THRESHOLD = 0.7
DEBOUNCE_WINDOWS = 3     # consecutive agreeing confident windows required to "trigger"


@dataclass
class TrialResult:
    label: int
    detected: bool
    latency_sec: float | None      # cue onset to the END of the window that completes the trigger
    false_trigger: bool            # a debounced trigger fired during pre-cue rest
    first_trigger_correct: bool    # the first debounced trigger after the cue was the right class
    n_rest_triggers: int
    n_rest_windows: int


@dataclass
class SimulationSummary:
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return np.mean([t.detected for t in self.trials])

    @property
    def first_trigger_accuracy(self) -> float:
        """Share of trials where the first trigger fired was correct. A real
        glove acts on the first trigger, so this is the safer headline."""
        return np.mean([t.first_trigger_correct for t in self.trials])

    @property
    def mean_latency_sec(self) -> float | None:
        lat = [t.latency_sec for t in self.trials if t.detected]
        return float(np.mean(lat)) if lat else None

    @property
    def median_latency_sec(self) -> float | None:
        lat = [t.latency_sec for t in self.trials if t.detected]
        return float(np.median(lat)) if lat else None

    @property
    def false_trigger_rate_per_trial(self) -> float:
        return np.mean([t.false_trigger for t in self.trials])

    @property
    def false_trigger_rate_per_window(self) -> float:
        total_windows = sum(t.n_rest_windows for t in self.trials)
        total_triggers = sum(t.n_rest_triggers for t in self.trials)
        return total_triggers / total_windows if total_windows else 0.0


def _slide_windows(raw_data: np.ndarray, sfreq: float, t_start: float, t_end: float,
                    window_sec: float, step_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """raw_data: (n_channels, n_times) around a fixed reference sample 0.
    Returns (windows: (n_windows, n_channels, window_samples), times: (n_windows,))
    where `times` is the window START time relative to the reference.
    """
    window_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    start_sample = int(round(t_start * sfreq))
    end_sample = int(round(t_end * sfreq)) - window_samples

    starts = np.arange(start_sample, max(start_sample, end_sample + 1), step_samples)
    windows = np.stack([raw_data[:, s:s + window_samples] for s in starts], axis=0)
    times = starts / sfreq
    return windows, times


def _confident_predictions(fbcsp: FBCSP, clf, windows: np.ndarray, gate: tuple | None = None):
    """Returns (preds, confident) for a batch of windows.

    If `gate` is given as (fbcsp_gate, clf_gate), a window only counts as
    "confident" when *both* the rest-vs-MI gate confidently says "MI present"
    and the left/right classifier is itself confident. That's a two-stage
    cascade rather than a single forced binary choice.
    """
    feats = fbcsp.transform(windows)
    proba = clf.predict_proba(feats)
    preds = proba.argmax(axis=1)
    confident = proba.max(axis=1) >= CONFIDENCE_THRESHOLD

    if gate is not None:
        fbcsp_gate, clf_gate = gate
        gate_feats = fbcsp_gate.transform(windows)
        gate_proba = clf_gate.predict_proba(gate_feats)
        gate_says_mi = gate_proba.argmax(axis=1) == 1  # gate label 1 == "MI present"
        gate_confident = gate_proba.max(axis=1) >= GATE_CONFIDENCE_THRESHOLD
        confident = confident & gate_says_mi & gate_confident

    return preds, confident


def _debounced_trigger(preds: np.ndarray, confident: np.ndarray) -> tuple[int | None, int | None]:
    """First index completing DEBOUNCE_WINDOWS consecutive confident windows
    that agree on one class. Returns (index, class), or (None, None)."""
    run_len, run_class = 0, None
    for i, (pred, ok) in enumerate(zip(preds, confident)):
        if not ok:
            run_len, run_class = 0, None
            continue
        run_len = run_len + 1 if pred == run_class else 1
        run_class = pred
        if run_len >= DEBOUNCE_WINDOWS:
            return i, int(pred)
    return None, None


def simulate_subject(
    fbcsp: FBCSP,
    clf,
    raw,
    events: np.ndarray,
    event_id: dict,
    label_map: dict,
    gate: tuple | None = None,
) -> SimulationSummary:
    """label_map: {event_code: 0_or_1} matching the classifier's training labels.

    gate: optional (fbcsp_gate, clf_gate) rest-vs-MI cascade stage; see
    `_confident_predictions`. When None, behaves as a plain forced binary
    left/right classifier with no concept of "rest."
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_channels, n_times)
    n_times = data.shape[1]

    summary = SimulationSummary()
    for sample, _, code in events:
        if code not in label_map:
            continue
        true_label = label_map[code]

        rest_windows, _ = _slide_windows(
            data[:, max(0, sample - int(PRE_CUE_SEC * sfreq)):sample], sfreq,
            t_start=0.0, t_end=PRE_CUE_SEC, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        ) if sample - int(PRE_CUE_SEC * sfreq) >= 0 else (np.empty((0,)), np.empty((0,)))

        post_end_sample = min(n_times, sample + int(POST_CUE_SEC * sfreq))
        post_windows, post_times = _slide_windows(
            data[:, sample:post_end_sample], sfreq,
            t_start=0.0, t_end=(post_end_sample - sample) / sfreq,
            window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        )

        n_rest_windows = rest_windows.shape[0] if rest_windows.ndim == 3 else 0
        n_rest_triggers = 0
        false_trigger = False
        if n_rest_windows > 0:
            preds, confident = _confident_predictions(fbcsp, clf, rest_windows, gate=gate)
            n_rest_triggers = int(confident.sum())
            false_trigger = _debounced_trigger(preds, confident)[0] is not None

        detected, latency, first_trigger_correct = False, None, False
        if post_windows.shape[0] > 0:
            preds, confident = _confident_predictions(fbcsp, clf, post_windows, gate=gate)
            first_idx, first_class = _debounced_trigger(preds, confident)
            first_trigger_correct = first_class == true_label

            correct_idx, _ = _debounced_trigger(preds, confident & (preds == true_label))
            if correct_idx is not None:
                detected = True
                # The decision exists only once the last window has been recorded.
                latency = float(post_times[correct_idx]) + WINDOW_SEC

        summary.trials.append(TrialResult(
            label=true_label,
            detected=detected,
            latency_sec=latency,
            false_trigger=false_trigger,
            first_trigger_correct=first_trigger_correct,
            n_rest_triggers=n_rest_triggers,
            n_rest_windows=n_rest_windows,
        ))

    return summary
