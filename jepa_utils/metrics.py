# utils/metrics.py
import torch
import numpy as np

def calculate_rmse(regression_weights, X, y, scaler_y):
    """Calculate RMSE on the validation data using the trained linear regression model.

    This helper includes several guards to prevent propagation of NaN/Inf values
    through the metric pipeline. If no valid samples are present or any tensor
    contains non-finite values, the metric is skipped and ``None`` is returned.
    """

    if regression_weights is None:
        return None

    # Ensure tensors share dtype and device with the regression weights
    X = X.to(regression_weights.device, dtype=regression_weights.dtype)
    y = y.to(regression_weights.device, dtype=regression_weights.dtype)

    # Skip metric if invalid values are detected
    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        print("Warning: metric skipped\u2014non-finite values detected in inputs")
        return None

    # Add bias term to X (since you added it during training)
    ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
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
    errors = (y_pred_exp - y_true_exp) ** 2
    if not np.isfinite(errors).all() or errors.size == 0:
        print("Warning: metric skipped\u2014invalid errors")
        return None

    mse = np.mean(errors)
    rmse = np.sqrt(mse)
    return rmse

