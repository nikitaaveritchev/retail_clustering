import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from retail_clustering.rfm import calculate_rfm


def test_calculate_rfm_basic():
    data = pd.DataFrame({
        "InvoiceDate": pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-04"]),
        "CustomerID": [1, 1, 2],
        "Quantity": [1, 2, 1],
        "Price": [10.0, 10.0, 20.0],
    })
    rfm = calculate_rfm(data, reference_date=pd.Timestamp("2021-01-05"))
    assert list(rfm.columns) == ["Recency", "Frequency", "Monetary"]
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 30.0
    assert rfm.loc[2, "Recency"] == 1
