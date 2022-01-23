import os

import numpy as np
import pandas as pd

from mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252


def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column close

    Returns:
        pd.DataFrame: input features
    """

    df_asset = df_asset[
        ~df_asset["close"].isna()
        | ~df_asset["close"].isnull()
        | (df_asset["close"] > 1e-8)  # price is zero
    ].copy()

    # winsorize using rolling 5X standard deviations to remove outliers
    df_asset["srs"] = df_asset["close"]
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    # vol scaling and shift to be next day returns
    df_asset["target_returns"] = calc_vol_scaled_returns(
        df_asset["daily_returns"], df_asset["daily_vol"]
    ).shift(-1)

    def calc_normalised_returns(day_offset):
        return (
            calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )

    df_asset["norm_daily_return"] = calc_normalised_returns(1)
    df_asset["norm_monthly_return"] = calc_normalised_returns(21)
    df_asset["norm_quarterly_return"] = calc_normalised_returns(63)
    df_asset["norm_biannual_return"] = calc_normalised_returns(126)
    df_asset["norm_annual_return"] = calc_normalised_returns(252)

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # date features
    if len(df_asset):
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        df_asset["week_of_year"] = df_asset.index.weekofyear
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []
        
    return df_asset.dropna()


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
