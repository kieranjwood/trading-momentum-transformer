import os

import numpy as np
import pandas as pd


def pull_quandl_sample_data(ticker: str) -> pd.DataFrame:
    return (
        pd.read_csv(os.path.join("data", "quandl", f"{ticker}.csv"), parse_dates=[0])
        .rename(columns={"Trade Date": "date", "Date": "date", "Settle": "close"})
        .set_index("date")
        .replace(0.0, np.nan)
    )
