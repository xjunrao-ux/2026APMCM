"""Apply the selected cascade to an acoustic CSV."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classification import existing
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "02_classification/models/cascade_50_stage3.joblib")
    args = parser.parse_args()
    rows = pd.read_csv(args.input, encoding="utf-8-sig")
    stages = existing.route(rows)
    rows["classification_stage"] = stages
    rows["predicted_CallType"] = np.select([stages.eq(1), stages.eq(2)], ["tv", "mod"], default="flat")
    stage3 = stages.eq(3)
    rows["stage3_score_mod"] = np.nan
    if stage3.any():
        model = joblib.load(args.model)
        score = existing.mod_score(model["stack"], rows.loc[stage3])
        rows.loc[stage3, "stage3_score_mod"] = score
        rows.loc[stage3, "predicted_CallType"] = np.where(score > model["threshold"], "mod", "flat")
    rows.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
