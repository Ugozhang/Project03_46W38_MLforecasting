""" Plot functions for forecaster """
import matplotlib.pyplot as plt
import numpy as np


def single_var_plot(x, y, xlabel=None, ylabel=None, title=None):
    """
    Simple timeseries plot for a single variable.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=ylabel or "value")
    plt.xlabel(xlabel or "Time")
    plt.ylabel(ylabel or "")
    plt.title(title or "")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def forecast_vs_real_plot(time, y_true, y_pred, title="Forecast vs Real", ylabel="Power"):
    """
    Plot predicted power output vs real measured power over time.

    Parameters
    ----------
    time : array-like
        Timestamps for the forecast horizon.
    y_true : array-like
        Real observed power values (e.g. one-hour-ahead).
    y_pred : array-like
        Predicted power values from the forecasting model.
    title : str
        Title of the plot.
    ylabel : str
        Label for the y-axis (e.g. "Power [kW]").
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time, y_true, label="Real power", linewidth=2)
    plt.plot(time, y_pred, label="Predicted power", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def multi_forecast_vs_real_plot(time, y_true, y_pred_dict, title="Forecast vs Real", ylabel="Power"):
    """
    Plot predicted power output vs real measured power over time for MULTIPLE models.

    Parameters
    ----------
    time : array-like
        Timestamps for the forecast horizon.
    y_true : array-like
        Real observed power values (e.g. one-hour-ahead).
    y_pred_dict : dict[str, array-like]
        Mapping from model name to its predicted power values.
    title : str
        Title of the plot.
    ylabel : str
        Label for the y-axis (e.g. "Power [kW]").
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time, y_true, label="Real power", linewidth=2)

    for model_name, y_pred in y_pred_dict.items():
        plt.plot(time, y_pred, label=f"{model_name} prediction", linewidth=1.8)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()