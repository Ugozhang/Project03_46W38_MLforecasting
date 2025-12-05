# wind_forecast/models.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from . import metrics

def make_lagged_features(df, target_col="Power", lags=1):
    """
    Build supervised dataset: predict target(t+1) from features at t.
    Simple version: just use Power(t) and windspeed_100m(t).
    """
    df = df.copy()
    df["target"] = df[target_col].shift(-1)   # Power(t+1)
    # Features at time t:
    features = ["Power", "windspeed_100m"]  # extend later
    valid = df.dropna(subset=["target"])    # drop last row (no t+1)
    X = valid[features].values
    y = valid["target"].values
    return X, y

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_mlp_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(train_df, test_df, feature_cols, target_col):
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    eva_metrics = metrics.ForecastEvaluator(y_test, y_pred)

    return model, {"mse": eva_metrics.mse(), "mae": eva_metrics.mae(), "rmse": eva_metrics.rmse()}