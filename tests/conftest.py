import numpy as np
import pandas as pd
import pytest

from src.data_loader import FEATURE_COLUMNS


@pytest.fixture
def sample_data():
    rows = []
    for index in range(80):
        churn = index % 2 == 0
        rows.append(
            {
                "customerID": f"fixture-{index}",
                "gender": "Female" if index % 3 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 4 else "No",
                "Dependents": "No" if index % 3 else "Yes",
                "tenure": 4 + index % 65,
                "PhoneService": "Yes",
                "MultipleLines": "No" if index % 2 else "Yes",
                "InternetService": ["DSL", "Fiber optic", "No"][index % 3],
                "OnlineSecurity": "Yes" if index % 2 else "No",
                "OnlineBackup": "No" if index % 2 else "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes" if index % 4 else "No",
                "StreamingTV": "Yes" if index % 3 else "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if churn else "One year",
                "PaperlessBilling": "Yes" if index % 2 else "No",
                "PaymentMethod": ["Electronic check", "Mailed check"][index % 2],
                "MonthlyCharges": float(30 + index % 100),
                "TotalCharges": float(100 + index * 10),
                "Churn": "Yes" if churn else "No",
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[0, "TotalCharges"] = np.nan
    return frame


@pytest.fixture
def feature_frame(sample_data):
    return sample_data.loc[:, FEATURE_COLUMNS].copy()
