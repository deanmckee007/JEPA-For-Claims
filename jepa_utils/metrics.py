# utils/metrics.py
import torch
import numpy as np

def calculate_rmse(regression_weights, X, y, scaler_y):
    """Calculate RMSE on the validation data using the trained linear regression model"""
    if regression_weights is None:
        return None
    
    regression_weights = regression_weights.to(X.device)

    # Add bias term to X (since you added it during training)
    ones = torch.ones(X.size(0), 1, device=X.device)
    X = torch.cat([X, ones], dim=1)

    # Predict using the validation data
    y_pred = X @ regression_weights  # Shape: [batch_size]

    # Move predictions and true values to CPU and convert to numpy
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y.cpu().numpy()

    # Inverse transform the predictions and targets using scaler_y
    y_pred_inv = scaler_y.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
    y_true_inv = scaler_y.inverse_transform(y_true_np.reshape(-1, 1)).flatten()

    # Since your target is ln(cost), apply exponential to get back to cost
    y_pred_inv = np.clip(y_pred_inv, a_min=None, a_max=10)
    y_pred_exp = np.exp(y_pred_inv)
    y_true_exp = np.exp(y_true_inv)

    # Compute RMSE in terms of actual cost
    mse = np.mean((y_pred_exp - y_true_exp) ** 2)
    rmse = np.sqrt(mse)
    return rmse

