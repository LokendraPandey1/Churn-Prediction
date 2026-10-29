import numpy as np
import pandas as pd
import pytest

from src.data_loader import FEATURE_COLUMNS


@pytest.fixture
def sample_data():
    rows = []
    for index in range(80):
        churn = index % 2
        rows.append(
            {
                "CustomerID": index + 1,
                "Age": 20 + index % 45,
                "Gender": "Female" if index % 3 else "Male",
                "Tenure": 1 + index % 60,
                "Usage Frequency": 1 + index % 30,
                "Support Calls": index % 11,
                "Payment Delay": index % 31,
                "Subscription Type": ["Basic", "Standard", "Premium"][index % 3],
                "Contract Length": ["Monthly", "Quarterly", "Annual"][index % 3],
                "Total Spend": 100 + index % 901,
                "Last Interaction": 1 + index % 30,
                "Churn": churn,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[0, "Total Spend"] = np.nan
    return frame


@pytest.fixture
def feature_frame(sample_data):
    return sample_data.loc[:, FEATURE_COLUMNS].copy()
