"""Pseudo-online simulation: train FBCSP+LDA on each subject's '0train'
session, then run a sliding-window decoder over the continuous '1test'
session to measure detection latency and false-trigger rate, the metrics
that determine whether this decoder could plausibly drive a physical glove
in real time, as opposed to just classifying pre-cut offline trials.

Runs two variants per subject:
  - baseline: plain forced binary left/right classifier (no concept of rest)
  - gated: adds a rest-vs-MI cascade stage trained on pre-cue EEG, so the
    decoder can decline to trigger when nothing is happening
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from bci_glove.data import load_rest_windows, load_subject_continuous, load_subject_epochs, subject_list
from bci_glove.fbcsp import FBCSP
from bci_glove.online_sim import simulate_subject

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "online_simulation.csv"


def _fit_lr_classifier(subject: int):
    X_train, y_train, sfreq = load_subject_epochs(subject, session="0train")
    fbcsp = FBCSP(sfreq=sfreq, n_csp_components=2, n_features_select=4)
    feats_train = fbcsp.fit_transform(X_train, y_train)
    clf = LinearDiscriminantAnalysis().fit(feats_train, y_train)
    return fbcsp, clf, sfreq


def _fit_rest_gate(subject: int, sfreq: float):
    X_mi, _, _ = load_subject_epochs(subject, session="0train")
    X_rest = load_rest_windows(subject, session="0train", n_samples=X_mi.shape[-1])
    X = np.concatenate([X_rest, X_mi], axis=0)
    y = np.concatenate([np.zeros(len(X_rest)), np.ones(len(X_mi))]).astype(int)  # 1 == "MI present"

    fbcsp_gate = FBCSP(sfreq=sfreq, n_csp_components=2, n_features_select=4)
    feats = fbcsp_gate.fit_transform(X, y)
    clf_gate = LinearDiscriminantAnalysis().fit(feats, y)
    return fbcsp_gate, clf_gate


def run_subject(subject: int) -> dict:
    fbcsp, clf, sfreq = _fit_lr_classifier(subject)
    fbcsp_gate, clf_gate = _fit_rest_gate(subject, sfreq)

    raw, events, event_id = load_subject_continuous(subject, session="1test")
    label_map = {code: (0 if name == "left_hand" else 1) for name, code in event_id.items()}

    baseline = simulate_subject(fbcsp, clf, raw, events, event_id, label_map, gate=None)
    gated = simulate_subject(fbcsp, clf, raw, events, event_id, label_map, gate=(fbcsp_gate, clf_gate))

    return {
        "subject": subject,
        "n_trials": len(baseline.trials),
        "baseline_detection_rate": baseline.detection_rate,
        "baseline_mean_latency_sec": baseline.mean_latency_sec,
        "baseline_false_trigger_rate": baseline.false_trigger_rate_per_trial,
        "gated_detection_rate": gated.detection_rate,
        "gated_mean_latency_sec": gated.mean_latency_sec,
        "gated_false_trigger_rate": gated.false_trigger_rate_per_trial,
    }


def main():
    rows = []
    for subject in subject_list():
        t0 = time.time()
        row = run_subject(subject)
        rows.append(row)
        print(
            f"subject={subject} "
            f"baseline: detect={row['baseline_detection_rate']:.2f} "
            f"false_trig={row['baseline_false_trigger_rate']:.2f} | "
            f"gated: detect={row['gated_detection_rate']:.2f} "
            f"false_trig={row['gated_false_trigger_rate']:.2f} "
            f"({time.time()-t0:.1f}s)"
        )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
