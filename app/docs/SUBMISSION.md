# Submission

Official outputs are `door_predictions.csv`, `acv_predictions.csv`, `rail_predictions.csv`, and
`shm_predictions.csv`, stored at the top level of `predictions.zip`. Door uses start/end/prediction;
ACV uses file_id/ranked_cars; rail and SHM use file_id/prediction. The validator enforces identifiers,
labels, ordering, types, and archive structure before packaging.

