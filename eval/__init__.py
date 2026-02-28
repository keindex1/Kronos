"""
Evaluation module for Kronos financial forecasting model.

This package provides comprehensive evaluation tools for assessing model performance
using both regression and classification metrics relevant to financial applications.
"""

from .evaluate import ModelEvaluator, evaluate_predictions, evaluate_multiple_assets

__all__ = [
    'ModelEvaluator',
    'evaluate_predictions', 
    'evaluate_multiple_assets'
]