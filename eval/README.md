# Kronos Evaluation Module

This module provides evaluation capabilities for the Kronos financial forecasting model. It includes both traditional sklearn-based evaluation and PyTorch-based evaluation implementations.

## Overview

The evaluation module offers comprehensive assessment of model performance for financial time series prediction tasks, with metrics specifically designed for financial applications:

- **Regression metrics**: MAE, MSE, RMSE, MAPE, R²
- **Financial metrics**: Information Coefficient (IC), Rank IC, Information Ratio (IR)
- **Classification metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Kappa

## Available Evaluators

### 1. Sklearn-based Evaluator (`evaluate.py`)
The original evaluator using sklearn for all computations.

### 2. PyTorch-based Evaluator (`evaluate_torch_optimized.py`)
An optimized evaluator using PyTorch tensors for computation, allowing for better integration with deep learning workflows and GPU acceleration.

## Usage

### Basic Usage with PyTorch Evaluator

```python
import torch
from evaluate_torch_optimized import evaluate_predictions

# Example actual and predicted values
actual = torch.tensor([100, 102, 101, 103, 105]).float()
predicted = torch.tensor([101, 103, 100, 104, 104]).float()

# Evaluate predictions
results = evaluate_predictions(
    y_true=actual,
    y_pred=predicted,
    print_results=True
)
```

### Multiple Assets Evaluation

```python
from evaluate_torch_optimized import evaluate_multiple_assets

assets_data = {
    'AAPL': (actual_aapl, predicted_aapl),
    'GOOGL': (actual_googl, predicted_googl),
}

all_results = evaluate_multiple_assets(
    asset_predictions=assets_data,
    print_results=True
)
```

### Direct Class Usage

```python
from evaluate_torch_optimized import TorchModelEvaluator

evaluator = TorchModelEvaluator()

# Calculate individual metrics
reg_metrics = evaluator.evaluate_regression(actual, predicted)
cls_metrics = evaluator.evaluate_classification(actual, predicted)
ic = evaluator.calculate_ic(actual, predicted)
```

### Basic Evaluation (Sklearn Version)

```python
from eval.model_evaluation import evaluate_predictions
import numpy as np

# Example with sample data
y_true = np.array([100, 102, 101, 105, 103])  # Actual prices
y_pred = np.array([101, 101, 103, 104, 102])  # Predicted prices

results = evaluate_predictions(y_true, y_pred)
```

## Key Features

- **Financial-specific metrics**: IC, Rank IC, and IR metrics specifically designed for financial modeling
- **Direction prediction**: Converts continuous predictions to binary up/down movements
- **Robust error handling**: Handles NaN values and edge cases gracefully
- **Batch processing**: Supports evaluation of multiple assets simultaneously
- **GPU compatibility**: PyTorch version can leverage GPU acceleration

## Performance Considerations

While the PyTorch evaluator provides better integration with deep learning workflows, note that for small datasets the sklearn version may be faster due to overhead in initializing PyTorch tensors. The PyTorch version becomes more beneficial when working with large datasets or when GPU acceleration is available.

## Integration with Kronos Model

To evaluate the Kronos model predictions:

```python
from model import Kronos, KronosTokenizer, KronosPredictor
from eval.model_evaluation import evaluate_predictions
import pandas as pd

# Load your model and tokenizer
tokenizer = KronosTokenizer.from_pretrained("pretrained/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("pretrained/Kronos-small")
predictor = KronosPredictor(model, tokenizer)

# Make predictions
# ... your prediction code ...

# Evaluate predictions
results = evaluate_predictions(actual_values, predicted_values)
```

## Purpose in Financial Context

- **Regression Metrics**: Essential for measuring the accuracy of price/volume forecasts
- **Classification Metrics**: Important for directional prediction (will price go up or down?)
- **Financial Metrics (IC/IR)**: Critical for quantifying the predictive power of factors and strategy effectiveness
- **Financial Applications**: Particularly useful for evaluating trading strategy effectiveness

The evaluation framework considers the unique challenges of financial data including high noise levels, non-stationarity, and the importance of directional accuracy over exact price prediction.