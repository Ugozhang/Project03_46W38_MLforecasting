#
import pandas as pd
import numpy as np

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw meteorological dataframe into ML-ready features 
    for wind turbine power prediction.

    Expected input columns:
        - Time
        - temperature_2m
        - relativehumidity_2m
        - dewpoint_2m
        - windspeed_10m
        - windspeed_100m
        - winddirection_10m
        - winddirection_100m
        - windgusts_10m
        - Power (target column, not modified here)

    Returns:
        A new DataFrame with engineered features such as:
        - cyclic time features (hour, day of year)
        - wind vector components (u, v)
        - vertical wind shear
    """
    # Work on a copy to avoid modifying the original dataframe
    df_ml = df.copy()

    # ------------------------------------------------------
    # 1. Time features
    # ------------------------------------------------------
    if "Time" in df_ml.columns:
        df_ml["Time"] = pd.to_datetime(df_ml["Time"])
        df_ml["hour"] = df_ml["Time"].dt.hour
        df_ml["dayofyear"] = df_ml["Time"].dt.dayofyear

        df_ml["hour_sin"] = np.sin(2 * np.pi * df_ml["hour"] / 24)
        df_ml["hour_cos"] = np.cos(2 * np.pi * df_ml["hour"] / 24)
        df_ml["doy_sin"] = np.sin(2 * np.pi * df_ml["dayofyear"] / 365.25)
        df_ml["doy_cos"] = np.cos(2 * np.pi * df_ml["dayofyear"] / 365.25)

        #df_ml = df_ml.drop(columns=["Time"])

    # ------------------------------------------------------
    # 2. Wind direction + wind speed → vector components (u, v)
    # ------------------------------------------------------
    for level in ("10m", "100m"):
        ws_col = f"windspeed_{level}"
        wd_col = f"winddirection_{level}"

        if ws_col in df_ml.columns and wd_col in df_ml.columns:
            theta = np.radians(df_ml[wd_col])
            df_ml[f"u_{level}"] = df_ml[ws_col] * np.cos(theta)
            df_ml[f"v_{level}"] = df_ml[ws_col] * np.sin(theta)

    # ------------------------------------------------------
    # 3. Vertical wind shear ()
    # ------------------------------------------------------
    if "windspeed_10m" in df.columns and "windspeed_100m" in df.columns:
        df_ml["delta_ws"] = df_ml["windspeed_100m"] - df_ml["windspeed_10m"]

    return df_ml



def train_test_split_time(df, test_fraction=0.2):
    """
    Split dataset into training and test groups, as lagged feature.
    """
    n = len(df)
    split_idx = int((1 - test_fraction) * n)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test