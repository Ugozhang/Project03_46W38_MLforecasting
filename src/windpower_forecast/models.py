""" ML models, and persistence model at the end. """
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


def _compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute basic regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    #rmse = root_mean_squared_error(y_true, y_pred)
    return {"mse": mse, "mae": mae, "rmse": rmse}

class BaseModelTrainer(ABC):
    """Abstract base class for ML model trainers with a unified train() method."""

    def __init__(self) -> None:
        self.model: Any | None = None
    
    @abstractmethod
    def build_model(self):
        """Return an uninitialized sklearn regressor."""
        raise NotImplementedError
    
    def train(self, train_df, test_df, feature_cols, target_col):
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        self.model = self.build_model()

        self.model.fit(X_train,y_train)

        y_pred = self.model.predict(X_test)

        scores = _compute_scores(y_test, y_pred)

        return self.model, scores, y_test, y_pred

class RandomForestTrainer(BaseModelTrainer):
    def build_model(self):
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    
class SVMTrainer(BaseModelTrainer):
    def build_model(self):
        return SVR(kernel="rbf", C=10.0, epsilon=0.01)

class MLPTrainer(BaseModelTrainer):
    def build_model(self):
        return MLPRegressor(
            hidden_layer_sizes=(64,32),
            activation="relu",
            max_iter=500,
            random_state=41,
        )

def persistence_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],  # unused, kept for API consistency
    target_col: str,
) -> tuple[None, dict, np.ndarray, np.ndarray]:
    """
    Persistence baseline: one-hour-ahead forecast.

    Assumes: y_true(t+1) = Power(t+1),
             y_pred(t+1) = Power(t).
    Here we assume the original current power is stored in column "Power".
    """
    if "Power" not in test_df.columns:
        raise ValueError("test_df must contain 'Power' column for persistence baseline.")
    
    y_true = test_df[target_col].values
    y_pred = test_df["Power"].values

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length for persistence baseline.")

    scores = _compute_scores(y_true, y_pred)
    return None, scores, y_true, y_pred


# Unified API functions for easy access from GUI
def train_random_forest(train_df, test_df, feature_cols, target_col):
    return RandomForestTrainer().train(train_df, test_df, feature_cols, target_col)

def train_svm(train_df, test_df, feature_cols, target_col):
    return SVMTrainer().train(train_df, test_df, feature_cols, target_col)

def train_mlp(train_df, test_df, feature_cols, target_col):
    return MLPTrainer().train(train_df, test_df, feature_cols, target_col)

# A lookup table for GUI/model-selection dropdown
MODEL_TRAINERS = {
    "Persistence baseline": persistence_baseline,
    "Random Forest": train_random_forest,
    "Support Vector Machine": train_svm,
    "Neural Network (MLP)": train_mlp,
    }
