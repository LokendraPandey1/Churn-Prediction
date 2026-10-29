# Model artifacts

Run `python -m src.train` after configuring a valid dataset. The command writes
the selected fitted pipeline to `models/churn_pipeline.joblib` and the
cross-validation and untouched test-set metrics to `models/metrics.json`.
These generated artifacts are intentionally ignored by Git. Never load a
Joblib artifact from an untrusted source.
