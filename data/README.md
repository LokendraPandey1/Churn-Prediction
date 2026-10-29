# Dataset setup

The application expects a CSV based on the IBM Telco Customer Churn schema,
including the 19 customer/service input columns and a `Churn` column containing
`Yes` or `No`. It does not download data automatically.

IBM documents its Cognos Telco churn sample as fictional and distributes it as
several Excel source files inside Cognos sample content. See IBM's
[Telco customer churn sample documentation](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn)
for access and licensing details. Export and join the relevant source data into
a CSV with the schema this project validates, or obtain the commonly used IBM
Telco CSV from a source whose terms you have checked. Do not treat the
fictional sample as real-world customer evidence.

Put the resulting file at `data/telco_churn.csv`, or set `CHURN_DATA_PATH` to an
absolute or project-relative CSV path. To train against another filename:

```powershell
$env:CHURN_DATA_PATH = "C:\path\to\telco_churn.csv"
python -m src.train
```

The loader never fabricates rows or downloads an unverified dataset. Dataset
records are excluded from Git by `.gitignore`.
