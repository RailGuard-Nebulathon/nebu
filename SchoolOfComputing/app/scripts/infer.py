"""Trusted-bundle batch inference CLI."""

import argparse
from pathlib import Path
import pandas as pd
from railguard.inference import RailGuardPredictor
from railguard.inference.batch import input_files
from railguard.constants import CORRUGATION_OFFICIAL, DOOR_OFFICIAL
from railguard.submission.schemas import OFFICIAL_FILES
from railguard.submission.validator import validate_prediction_frame


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--task",required=True,choices=("door","acv","corrugation","shm")); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path); args=parser.parse_args()
    predictor=RailGuardPredictor.from_bundle(args.checkpoint); per_file=[]
    for path in input_files(args.input): per_file.append((path, predictor.predict(args.task,path)))
    if args.task == "door":
        rows=[{"start_time":item.metadata["start_time"],"end_time":item.metadata["end_time"],"prediction":DOOR_OFFICIAL.get(str(item.prediction),str(item.prediction))} for _,items in per_file for item in items]
    elif args.task == "acv":
        rows=[]
        for path,items in per_file:
            score=lambda item: (item.probabilities or {}).get("True",(item.probabilities or {}).get("1",item.confidence or 0.0))
            ranking=sorted(items,key=score,reverse=True)
            rows.append({"file_id":path.name,"ranked_cars":"|".join(item.sample_id for item in ranking)})
    elif args.task == "corrugation":
        rows=[{"file_id":path.name,"prediction":CORRUGATION_OFFICIAL.get(str(items[0].prediction),str(items[0].prediction))} for path,items in per_file]
    else:
        rows=[{"file_id":path.name,"prediction":float(items[0].prediction)} for path,items in per_file]
    frame=pd.DataFrame(rows); validate_prediction_frame(args.task,frame)
    output=args.output or Path("outputs/predictions")/OFFICIAL_FILES[args.task]
    output.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(output,index=False); print(output)


if __name__ == "__main__": main()
