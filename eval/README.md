# Model Evaluation

This module contains comprehensive evaluation tools for assessing the performance of the Kronos financial forecasting model. It includes both regression and classification metrics commonly used in financial prediction tasks.

## Features

### Regression Metrics
- **MAE (Mean Absolute Error)**: Measures average magnitude of errors without considering direction
- **MSE (Mean Squared Error)**: Penalizes larger errors more heavily than smaller ones
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the target variable
- **MAPE (Mean Absolute Percentage Error)**: Relative error expressed as a percentage
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

### Classification Metrics
- **Accuracy**: Overall proportion of correct predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **Kappa Coefficient**: Adjusts accuracy by accounting for agreement occurring by chance

## Usage

### Basic Evaluation
```python
from eval.model_evaluation import evaluate_predictions
import numpy as np

# Example with sample data
y_true = np.array([100, 102, 101, 105, 103])  # Actual prices
y_pred = np.array([101, 101, 103, 104, 102])  # Predicted prices

results = evaluate_predictions(y_true, y_pred)
```

### Multiple Assets Evaluation
```python
from eval.model_evaluation import evaluate_multiple_assets

assets = {
    'AAPL': (actual_aapl, predicted_aapl),
    'GOOGL': (actual_googl, predicted_googl),
    'MSFT': (actual_msft, predicted_msft)
}

multi_results = evaluate_multiple_assets(assets)
```

### Using the Evaluator Class Directly
```python
from eval.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Get regression metrics only
reg_metrics = evaluator.evaluate_regression(y_true, y_pred)

# Get classification metrics only
clf_metrics = evaluator.evaluate_classification(y_true, y_pred)

# Get all metrics
all_metrics = evaluator.evaluate_all(y_true, y_pred)
```

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
- **Financial Applications**: Particularly useful for evaluating trading strategy effectiveness

The evaluation framework considers the unique challenges of financial data including high noise levels, non-stationarity, and the importance of directional accuracy over exact price prediction.