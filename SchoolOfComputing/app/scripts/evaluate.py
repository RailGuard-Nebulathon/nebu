"""Evaluate saved truth/prediction CSVs at sample level."""

import argparse, json
from pathlib import Path
import pandas as pd
from railguard.evaluation import classification_metrics, regression_metrics


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--task",required=True,choices=("door","acv","corrugation","shm")); parser.add_argument("--checkpoint",type=Path); parser.add_argument("--split",default="validation"); parser.add_argument("--truth",type=Path); parser.add_argument("--predictions",type=Path); args=parser.parse_args()
    if not args.truth or not args.predictions: raise SystemExit("Provide --truth and --predictions generated from an inspected evaluation split")
    truth,pred=pd.read_csv(args.truth),pd.read_csv(args.predictions); merged=truth.merge(pred,on="file_id",suffixes=("_true","_pred"),validate="one_to_one")
    result=regression_metrics(merged.prediction_true,merged.prediction_pred) if args.task=="shm" else classification_metrics(merged.prediction_true,merged.prediction_pred)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()

