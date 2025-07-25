from __future__ import annotations

import pandas as pd


def calculate_rfm(
    df: pd.DataFrame,
    invoice_date_col: str = "InvoiceDate",
    customer_id_col: str = "CustomerID",
    quantity_col: str = "Quantity",
    price_col: str = "Price",
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return RFM metrics from the dataset."""
    df = df.copy()
    df["TotalPrice"] = df[quantity_col] * df[price_col]
    if reference_date is None:
        reference_date = df[invoice_date_col].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(customer_id_col)
        .agg({
            invoice_date_col: lambda x: (reference_date - x.max()).days,
            quantity_col: "count",
            "TotalPrice": "sum",
        })
        .rename(columns={
            invoice_date_col: "Recency",
            quantity_col: "Frequency",
            "TotalPrice": "Monetary",
        })
    )

    return rfm
