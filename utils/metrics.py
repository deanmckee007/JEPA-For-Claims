# utils/metrics.py
import torch

def calculate_rmse(regression_weights, X, y):
    """Calculate RMSE on the validation data using the trained linear regression model"""
    if regression_weights is None:
        return None

    # Predict using the validation data
    y_pred = X @ regression_weights
    mse = torch.mean((y_pred - y) ** 2)
    rmse = torch.sqrt(mse)
    return rmse
