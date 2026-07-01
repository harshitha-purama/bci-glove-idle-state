"""Generates results/online_simulation_comparison.png: baseline vs. gated
detection rate and false-trigger rate, per subject.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    rows = list(csv.DictReader(open(RESULTS_DIR / "online_simulation.csv")))
    subjects = [int(r["subject"]) for r in rows]
    x = np.arange(len(subjects))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x - width / 2, [float(r["baseline_false_trigger_rate"]) for r in rows], width, label="baseline (no gate)", color="tab:red")
    ax.bar(x + width / 2, [float(r["gated_false_trigger_rate"]) for r in rows], width, label="gated (rest-vs-MI)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{s:02d}" for s in subjects])
    ax.set_ylabel("False-trigger rate (fraction of trials)")
    ax.set_title("False triggers during rest: before vs. after gating")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, [float(r["baseline_detection_rate"]) for r in rows], width, label="baseline (no gate)", color="tab:red")
    ax.bar(x + width / 2, [float(r["gated_detection_rate"]) for r in rows], width, label="gated (rest-vs-MI)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{s:02d}" for s in subjects])
    ax.set_ylabel("Detection rate (fraction of trials)")
    ax.set_title("True-positive detection: before vs. after gating")
    ax.legend()

    fig.suptitle("Effect of a rest-vs-MI gate on pseudo-online BCI performance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "online_simulation_comparison.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'online_simulation_comparison.png'}")

    means = {
        "baseline_detection_rate": np.mean([float(r["baseline_detection_rate"]) for r in rows]),
        "gated_detection_rate": np.mean([float(r["gated_detection_rate"]) for r in rows]),
        "baseline_false_trigger_rate": np.mean([float(r["baseline_false_trigger_rate"]) for r in rows]),
        "gated_false_trigger_rate": np.mean([float(r["gated_false_trigger_rate"]) for r in rows]),
        "baseline_mean_latency_sec": np.mean([float(r["baseline_mean_latency_sec"]) for r in rows]),
        "gated_mean_latency_sec": np.mean([float(r["gated_mean_latency_sec"]) for r in rows]),
    }
    print("Means across subjects:", {k: round(v, 3) for k, v in means.items()})


if __name__ == "__main__":
    main()
