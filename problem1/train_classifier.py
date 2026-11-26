"""Run the five stages in the handover order, from raw CSVs to verified results."""
from acoustics import OUT, audit
from classifiers import classify
from explore import explore, tests
from integration import conclude


def main():
    print("[A] Auditing Data01 ...", flush=True)
    data = audit()
    print("[B] Descriptive statistics and figures ...", flush=True)
    desc = explore(data)
    print("[C] Rank tests, effect sizes and within-date sensitivity ...", flush=True)
    comparisons = tests(data)
    print("[D] Six common features: L2, RBF-SVM, random forest. Nested CV ...", flush=True)
    summary, classes, coefficients = classify(data)
    print("[E] Connecting propagation and playback evidence ...", flush=True)
    conclude(desc, comparisons, summary, classes, coefficients)
    print(summary.iloc[:, :5].round(3).to_string(index=False))
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
