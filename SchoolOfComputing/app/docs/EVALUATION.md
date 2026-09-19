# Evaluation

Splits occur at segment/file/case level, never at correlated row level. ACV uses leave-one-case-out;
corrugation uses stratification unless grouping metadata is discovered; SHM uses file-level folds;
Door stratifies segment status and operation. Primary official metrics are Door IoU-weighted F1,
ACV linear rank decay, corrugation macro F1, and SHM `max(0, 1-MAPE)`. Preprocessing is fit in-fold.

