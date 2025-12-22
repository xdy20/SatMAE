"""
评估工具
"""
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, classification_report
)
import numpy as np


def evaluate_regression(y_true, y_pred):
    """
    统一的回归评估

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        metrics: 字典，包含R2, RMSE, MAE, MAPE
    """
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }


def evaluate_classification(y_true, y_pred):
    """
    统一的分类评估

    参数:
        y_true: 真实标签
        y_pred: 预测标签

    返回:
        metrics: 字典，包含Accuracy, F1_Macro, F1_Weighted
        report: 详细分类报告
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1_Macro': f1_score(y_true, y_pred, average='macro'),
        'F1_Weighted': f1_score(y_true, y_pred, average='weighted')
    }

    report = classification_report(y_true, y_pred)

    return metrics, report
