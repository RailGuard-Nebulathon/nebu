"""Fit split-conformal residual intervals from held-out CSV predictions."""

import argparse, json
from pathlib import Path
import pandas as pd
from railguard.calibration.conformal import SplitConformalRegressor


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--calibration",type=Path,required=True,help="CSV with truth,prediction"); parser.add_argument("--coverage",type=float,default=.9); parser.add_argument("--output",type=Path,default=Path("outputs/reports/conformal.json")); args=parser.parse_args()
    frame=pd.read_csv(args.calibration); model=SplitConformalRegressor(args.coverage).fit(frame.truth.to_numpy(),frame.prediction.to_numpy()); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({"coverage":args.coverage,"quantile":model.quantile},indent=2)); print(args.output)


if __name__ == "__main__": main()

