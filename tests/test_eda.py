from src.eda import build_eda_report


def test_eda_report_is_derived_from_supplied_dataset(sample_data):
    report = build_eda_report(sample_data)
    assert report["summary"]["shape"] == [len(sample_data), sample_data.shape[1]]
    assert report["summary"]["class_distribution"]["Churned"]["count"] == 40
    assert report["summary"]["duplicates"] == 0
    assert set(report["segment_churn_rates"]) == {
        "Contract Length",
        "Tenure",
        "Subscription Type",
        "Gender",
        "Support Calls",
        "Payment Delay",
    }
