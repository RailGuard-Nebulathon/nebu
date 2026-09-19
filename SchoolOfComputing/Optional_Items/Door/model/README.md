# Door trained bundle

The source Door artifacts arrived with shifted filenames and were normalized:

| Source | Packaged | Content |
|---|---|---|
| `config.yaml` | `metadata.json` | model metadata |
| `feature_names.json` | `config.yaml` | selection configuration |
| `cv_report.json` | `schema.json` | feature schema |
| `schema.json` | `model.joblib` | serialized `DoorEnsemble` |
| `model.joblib` | `preprocessor.joblib` | preprocessor |

`feature_names.json` was extracted from the supplied schema's feature array.
The absent detailed fold report was not invented. Metadata records five-fold
operation/status-stratified classification macro F1 of 1.0000 on labelled
segments, not end-to-end IoU-weighted F1. Load Joblib only from this trusted
package in a compatible scikit-learn environment.

