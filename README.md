# Customer Churn Prediction & Retention Analytics

A Python and Streamlit application for exploring the customer churn datasets in
`data/`, comparing classifiers, scoring individual or batch records, and
reviewing observed churn segments. Data validation, exploratory analysis,
preprocessing, training, scoring, and dashboard code are separated into
maintainable modules.

## Dataset in this checkout

The project is configured for the supplied
`customer_churn_dataset-training-master.csv` and
`customer_churn_dataset-testing-master.csv` files. Inspection found 440,833
training rows including one wholly blank row, which the loader discards, and
64,374 testing rows. The target is numeric `Churn` (`0` retained, `1` churned).
The training file has 249,999 churned and 190,833 retained labeled rows; the
testing file has 30,493 churned and 33,881 retained rows. These are counts from
the supplied CSVs, not claims about a real-world population.

Inputs are `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`,
`Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, and
`Last Interaction`. `CustomerID` is excluded from all model features. The two
files reuse some ID values for different records; the IDs are file-local
identifiers, not cross-file join keys.

The data provider and license were not included with the supplied files. Check
their terms and attribution before redistribution. See [data/README.md](data/README.md)
for file handling and configurable paths.

## Features

- Validated loading for the supplied numeric-label churn schema.
- Interactive overview, filters, churn segment comparisons, distributions, and
  correlations using Streamlit and Plotly.
- Logistic Regression, Random Forest, and Gradient Boosting model comparison.
- Preprocessing learned only inside scikit-learn training pipelines.
- Stratified cross-validation on the training file and evaluation against the
  supplied separate testing file.
- Saved fitted pipeline and measured CV/test metrics.
- ROC curve, confusion matrix, precision, recall, F1, ROC-AUC, permutation and
  tree feature importance, and local model-sensitivity explanations.
- Single-customer form and optional batch CSV scoring with downloadable output.
- Retention segment summaries and an optional user-assumption-based financial
  scenario.

Predictions and explanations are estimates of associations in the data. They
do not prove why customers churn or that a suggested retention intervention
will prevent churn.

## Setup (Windows PowerShell)

Python 3.11 or newer is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied files are the defaults. To use different files:

```powershell
$env:CHURN_DATA_PATH = "C:\path\to\training.csv"
$env:CHURN_TEST_DATA_PATH = "C:\path\to\testing.csv"
```

These environment-variable paths apply to the current terminal session.

## Run EDA, train, and launch

From the project root:

```powershell
python -m src.eda
python -m src.train
streamlit run app.py
```

The EDA command writes data-derived descriptive statistics, data quality,
churn distribution, numeric correlations, and segment rates to
`reports/eda_summary.json`.

Training writes:

- `models/churn_pipeline.joblib`: selected fitted preprocessing and classifier,
  feature schema, threshold, and a generic reference profile.
- `models/metrics.json`: cross-validation results for each candidate and
  holdout evaluation results from the supplied testing CSV.

Specify files or CV folds explicitly when needed:

```powershell
python -m src.train --data "C:\path\to\training.csv" --test-data "C:\path\to\testing.csv" --cv-folds 5
```

If `--test-data` points to a nonexistent file, training falls back to a
stratified 20% holdout from the training CSV.

## Evaluation methodology

1. The loader validates required columns, values, and numeric types, drops only
   wholly blank rows, and rejects rows with missing targets.
2. `CustomerID` and `Churn` are excluded from model features. Numeric median
   imputation/scaling and categorical imputation/one-hot encoding are fitted
   within each model's training pipeline.
3. Candidate models are compared using stratified cross-validation on the
   training CSV. Model selection uses mean ROC-AUC, with mean recall used only
   to break ties.
4. The selected model is evaluated against the separate supplied testing CSV,
   which is not used in training or model selection. When no separate test file
   is configured, an internal stratified 20% holdout is used instead.
5. Each candidate is also scored against that holdout, to make any
   cross-validation/holdout generalization gap visible. Holdout metrics include
   precision, recall, F1, ROC-AUC, classification report, confusion matrix,
   and ROC data, at the baseline threshold of 0.50. The test file's churn share
   differs from the training file's share; metrics should be interpreted in
   that context. The dashboard warns when selected-model CV ROC-AUC exceeds
   holdout ROC-AUC by 0.10 or more.

Recall measures how many churners are detected. Precision measures how many
flagged customers actually churned. Increasing recall can raise the number of
false positives, so any operational threshold should be chosen using validation
data and intervention costs—not the final evaluation set.

## Measured results for the supplied files

Results below were generated by `python -m src.train` using five-fold
stratified CV on the 440,832 usable training rows and the separate 64,374-row
testing CSV. Precision/recall/F1 on the test file use threshold 0.50.

| Model | CV ROC-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.960 | 0.689 | 0.531 | 0.988 | 0.691 |
| Random Forest | 1.000 | 0.616 | 0.488 | 0.999 | 0.656 |
| Gradient Boosting | 1.000 | 0.723 | 0.490 | 0.998 | 0.657 |

Random Forest is selected according to the predeclared CV ROC-AUC rule, but
its supplied-test ROC-AUC is only 0.616. The training churn proportion is
56.7%, versus 47.4% in the testing file, and every model shows a substantial
CV-to-test performance gap. This indicates poor generalization and/or a
material difference in data generation or distribution. Do not treat the
selected model as ready for operational decisions; verify the data source,
target definitions, and sampling process, then validate on a representative
holdout before relying on predictions. The test data has not been used to
change the model selection rule or threshold.

## Tests

```powershell
python -m pytest
```

The automated tests use small in-memory fixtures to verify the pipeline and
do not replace or alter the supplied customer datasets.

## Limitations

- The provider and licensing details for the supplied CSVs are unknown.
- Dataset patterns and measured model quality may not generalize to other
  populations or time periods.
- The overlapping `CustomerID` values are not consistent record identifiers
  across the two files; they are excluded from model inputs and not used for
  joining.
- The default threshold is a baseline, not an economically optimized cutoff.
- Permutation importance, tree importance, and local probability changes
  describe model behavior; none establish causality.
- No experimental evidence establishes retention-strategy effectiveness. The
  financial scenario is hypothetical and depends on user-supplied assumptions.
- The project does not include authentication, customer database storage,
  monitoring, or production deployment safeguards.

## Future improvements

- Validate on a clearly documented, time-separated sample and monitor drift.
- Calibrate probabilities and choose thresholds using explicit intervention
  costs and validation data.
- Test retention interventions with randomized experiments before estimating
  causal impact.
- Add access controls, privacy review, and monitoring before any operational
  deployment.
