# Supplied customer churn datasets

The application now uses the two CSV files supplied in this directory:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

They use the same schema. The model inputs are `Age`, `Gender`, `Tenure`,
`Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`,
`Contract Length`, `Total Spend`, and `Last Interaction`. `Churn` is a numeric
0/1 target. `CustomerID` is excluded from all model inputs.

The training file has one wholly blank row; the loader drops wholly blank rows
and rejects partially populated rows with a missing target. The test file is
used as the separate holdout evaluation set by default. `CustomerID` values
overlap between the supplied files, but the records for shared IDs differ, so
IDs are treated as file-local labels and are neither model features nor join
keys.

The dataset source and license were not included with the files. Check the
provider's attribution and terms before redistribution or making claims about
real-world customers. Results describe these supplied data only.

The CSVs are excluded from Git. To use different files:

```powershell
$env:CHURN_DATA_PATH = "C:\path\to\training.csv"
$env:CHURN_TEST_DATA_PATH = "C:\path\to\testing.csv"
python -m src.eda
python -m src.train
```

Or specify the paths directly:

```powershell
python -m src.train --data "C:\path\to\training.csv" --test-data "C:\path\to\testing.csv"
```
