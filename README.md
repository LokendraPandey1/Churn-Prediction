# Customer Churn Prediction & Retention Analytics

A Python and Streamlit application for exploring customer churn, training and
comparing churn classifiers, scoring individual or batch records, and reviewing
observed high-churn segments. The implementation separates data science logic
from the dashboard and uses a reproducible scikit-learn preprocessing/model
pipeline.

> **Dataset status:** No customer dataset was present in the repository when
> this project was implemented. The repository does not contain trained model
> results or claim measured findings. Supply a valid CSV and run the training
> command below to generate genuine EDA and evaluation results. The automated
> tests use small in-memory fixtures only; they are not presented as customer
> data or product metrics.

## Problem and intended use

The project estimates whether a customer is likely to churn based on their
current demographic, account, service, and billing attributes. It offers
descriptive analysis and predictive estimates to support further investigation.
It is a student portfolio project, not a production decision system.

Predicted probabilities and feature explanations describe model associations.
They do **not** prove why a customer churned or that a suggested retention action
will prevent churn. Test-set metrics are kept separate from cross-validation
results, and the final test set is not used to select the model or tune its
threshold.

## Features

- Validated CSV ingestion, with configurable path and no automatic downloads.
- Interactive overview, filters, churn segment comparisons, distributions, and
  correlations in Streamlit/Plotly.
- Three comparable models: class-weighted Logistic Regression, class-weighted
  Random Forest, and Gradient Boosting.
- Stratified cross-validation on the training partition and evaluation on an
  untouched stratified holdout set.
- Persisted fitted preprocessing/classifier pipeline and JSON metrics.
- ROC curve, confusion matrix, precision, recall, F1, ROC-AUC, permutation
  importance, supported tree importance, and local model-sensitivity view.
- Single-customer form and optional CSV batch scoring with downloadable output.
- Retention segment summaries and an optional financial scenario whose inputs
  are explicitly user-supplied assumptions.
- Pytest coverage of validation, preprocessing, training, artifact creation,
  and scoring.

## Data source, schema, and licensing

The initial schema follows the commonly used IBM Telco Customer Churn CSV. IBM
documents a fictional Telco churn sample in its
[Cognos Analytics sample documentation](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn).
IBM's documented sample is distributed as multiple Excel source files inside
Cognos sample content, rather than as a ready-to-use CSV at a public download
URL. Check the source's terms and licensing before obtaining, transforming, or
redistributing any data. See [data/README.md](data/README.md) for setup notes.

Expected features are `gender`, `SeniorCitizen`, `Partner`, `Dependents`,
`tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
`StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
`MonthlyCharges`, and `TotalCharges`. Training additionally requires a `Churn`
label with `Yes`/`No` values. A `customerID` column is allowed but excluded from
the model. Extra columns are ignored for model fitting; the target is rejected
as a prediction input. Blank numeric values are treated as missing and imputed
inside the training pipeline. Exact duplicate rows are reported in the
dashboard and not silently removed.

## Setup

Python 3.11 or newer is recommended. The requirements are compatible with
current stable Python versions supported by the selected dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the prepared CSV at `data/telco_churn.csv`. Alternatively, configure a
different local path:

```powershell
$env:CHURN_DATA_PATH = "C:\path\to\telco_churn.csv"
```

The path setting is per terminal session. Data files and generated models are
excluded from Git.

## Run EDA, train, and launch

From the project root, after setting up the dataset:

```powershell
python -m src.eda
python -m src.train
streamlit run app.py
```

Training writes:

- `models/churn_pipeline.joblib`: selected fitted preprocessing and classifier,
  fixed feature schema, threshold, and a non-customer-specific reference profile.
- `models/metrics.json`: per-model CV means and standard deviations; selected
  model; final holdout metrics, confusion matrix, classification report, ROC
  curve coordinates, and test-set feature importances.

Override paths or folds as needed:

```powershell
python -m src.train --data "C:\path\to\telco_churn.csv" --artifacts-dir models --cv-folds 5
```

EDA writes `reports/eda_summary.json` by default, with actual dataset shape,
descriptive statistics, missingness, duplicate count, churn distribution, and
segment rates.

## Evaluation methodology

1. The schema validator separates the fixed feature columns from the `Churn`
   target and excludes `customerID`, labels, and unrelated extra fields.
2. A stratified 80/20 split reserves the final test set before fitting any
   preprocessing transforms.
3. Each candidate pipeline learns numeric median imputation/scaling and
   categorical imputation/one-hot encoding within stratified cross-validation
   folds on the training partition only.
4. Model selection uses mean cross-validation ROC-AUC, with mean recall used
   only to break ties. Accuracy is not the selection metric.
5. The selected candidate is fitted on training data and evaluated once on the
   untouched holdout at the documented 0.50 threshold. Recall, precision, F1,
   ROC-AUC, classification report, confusion matrix, and ROC data are saved.
6. Recall measures how many churners are detected; precision measures how many
   flagged customers actually churned. Raising recall can produce more false
   positives, so businesses should choose intervention thresholds using
   validation data and actual intervention costs—not the final test set.

## Current results

No model results are published in this checkout because no dataset was
available to train on. After training, the application reads the measured
results directly from `models/metrics.json`. No metrics or customer insights
are hard-coded.

## Screenshots

Screenshots will be added after a real dataset has been configured and the
application has been run. No placeholder screenshots are represented as real
application output.

## Tests

```powershell
python -m pytest
```

Tests create an explicitly labeled in-memory fixture for functionality checks;
that fixture is not used by the app or reported as a real dataset.

## Limitations

- The initial data validator supports the classic IBM-style Telco schema. Other
  churn schemas need a deliberate mapping and validation update.
- IBM's documented sample is fictional and may not represent a production
  telecom population.
- Performance depends on data quality, population drift, label definition, and
  the train/test split. A single holdout is not a production guarantee.
- The default 0.50 classification threshold is a baseline. No operating
  threshold has been optimized against business costs.
- Feature importance and local probability changes describe model behavior,
  not causality.
- No experimental evidence establishes the effectiveness of suggested
  retention strategies. The financial scenario is hypothetical and requires
  user-entered assumptions.
- The project does not include authentication, customer database storage,
  monitoring, or a production deployment configuration.

## Future improvements

- Validate on temporally separated and out-of-period customer data.
- Calibrate probabilities and choose thresholds using explicit intervention
  costs and validation data.
- Add monitored deployment, access controls, privacy review, and drift checks
  before any operational use.
- Test retention interventions with randomized experiments before estimating
  causal impact.
