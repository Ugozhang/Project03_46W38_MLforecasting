""" Plot functions for forecaster """
import matplotlib.pyplot as plt
import numpy as np

def single_var_plot(x, y, xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def forecast_vs_real_plot(time, y_true, y_pred, title="Forecast vs Real"):
    """
    Plot predicted power output vs real measured power over time.

    Inputs:
        time  : pandas Series or array of timestamps
        y_true: real observed power values
        y_pred: predicted one-hour-ahead power values
        title : title of the plot

    This function is used for:
        - persistence baseline forecast visualization
        - ML model forecast visualization (SVM, Random Forest, MLP, etc.)
    """

    plt.figure(figsize=(12,5))
    plt.plot(time, y_true, label="Real Power", linewidth=2)
    plt.plot(time, y_pred, label="Predicted Power", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Power [kW]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()