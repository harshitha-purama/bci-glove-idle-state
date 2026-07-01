"""Generates results/accuracy_by_subject.png from results/benchmark.csv."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    data = defaultdict(dict)  # data[classifier][subject] = acc
    with open(RESULTS_DIR / "benchmark.csv") as f:
        for row in csv.DictReader(f):
            data[row["classifier"]][int(row["subject"])] = float(row["mean_accuracy"])

    subjects = sorted(next(iter(data.values())).keys())
    classifiers = sorted(data.keys())
    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, clf in enumerate(classifiers):
        accs = [data[clf][s] for s in subjects]
        ax.bar(x + (i - 1) * width, accs, width, label=clf.upper())

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{s:02d}" for s in subjects])
    ax.set_ylabel("10-fold CV accuracy")
    ax.set_xlabel("Subject (BCI Competition IV-2a)")
    ax.set_title("FBCSP: left-hand vs. right-hand motor-imagery classification")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_by_subject.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'accuracy_by_subject.png'}")

    means = {clf: np.mean(list(data[clf].values())) for clf in classifiers}
    print("Mean accuracy across subjects:", {k: round(v, 4) for k, v in means.items()})


if __name__ == "__main__":
    main()
