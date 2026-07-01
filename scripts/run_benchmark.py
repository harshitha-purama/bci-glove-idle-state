"""Runs FBCSP cross-validated accuracy across all 9 subjects of BCI IV-2a
and writes results/benchmark.csv for comparison against published numbers.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

from bci_glove.data import load_subject_epochs, subject_list
from bci_glove.evaluate import cross_validate_subject

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "benchmark.csv"


def main():
    rows = []
    for subject in subject_list():
        t0 = time.time()
        X, y, sfreq = load_subject_epochs(subject)
        for clf in ["gnb", "lda", "svm"]:
            res = cross_validate_subject(X, y, sfreq, classifier=clf, n_splits=10)
            rows.append({
                "subject": subject,
                "classifier": clf,
                "mean_accuracy": res["mean_accuracy"],
                "std_accuracy": res["std_accuracy"],
            })
            print(f"subject={subject} clf={clf} acc={res['mean_accuracy']:.3f} "
                  f"std={res['std_accuracy']:.3f} ({time.time()-t0:.1f}s elapsed)")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "classifier", "mean_accuracy", "std_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
