"""
Regression metrics for ablation study.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # sklearn removed mean_squared_error(..., squared=False) in 1.4+;
    # compute RMSE directly so this works on both old and new sklearn.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }
