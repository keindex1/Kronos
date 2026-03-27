import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TorchModelEvaluator:
    """
    A comprehensive evaluator for financial forecasting models that computes
    regression metrics for stock price prediction using PyTorch.
    Optimized for better performance with additional metrics.
    """

    def __init__(self):
        """Initialize the evaluator."""
        pass

    @staticmethod
    def calculate_returns(prices: torch.Tensor) -> torch.Tensor:
        """
        Calculate log returns from prices.
        
        Args:
            prices: Tensor of prices
            
        Returns:
            Tensor of log returns
        """
        # Calculate log returns: log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
        log_prices = torch.log(prices)
        returns = torch.diff(log_prices)
        return returns

    def calculate_mae(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Mean Absolute Error"""
        return torch.mean(torch.abs(y_true - y_pred))

    def calculate_mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Mean Squared Error"""
        return torch.mean((y_true - y_pred) ** 2)

    def calculate_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Root Mean Squared Error"""
        return torch.sqrt(self.calculate_mse(y_true, y_pred))

    def calculate_mape(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = torch.abs(y_true) > 1e-8
        if not torch.any(mask):
            return torch.tensor(float('nan'), dtype=y_true.dtype)
        return torch.mean(torch.abs((y_true - y_pred) / y_true) * mask.float()) * 100

    def calculate_r2(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate R-squared score"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        if ss_tot == 0:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
        return 1 - (ss_res / ss_tot)

    def calculate_ic(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate Information Coefficient (IC) - Pearson correlation between actual and predicted values.
        
        Args:
            y_true: Actual values tensor
            y_pred: Predicted values tensor
            
        Returns:
            Pearson correlation coefficient tensor
        """
        # Flatten inputs to ensure 1D arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Check if tensors are too small
        if len(y_true_flat) < 2:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
            
        # Calculate Pearson correlation coefficient
        mean_true = torch.mean(y_true_flat)
        mean_pred = torch.mean(y_pred_flat)
        
        numerator = torch.sum((y_true_flat - mean_true) * (y_pred_flat - mean_pred))
        denominator = torch.sqrt(torch.sum((y_true_flat - mean_true) ** 2) * torch.sum((y_pred_flat - mean_pred) ** 2))
        
        if denominator == 0:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
        
        ic = numerator / denominator
        return ic if not torch.isnan(ic) else torch.tensor(0.0, dtype=y_true.dtype)

    def calculate_rank_ic(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate Rank Information Coefficient (Spearman correlation).
        
        Args:
            y_true: Actual values tensor
            y_pred: Predicted values tensor
            
        Returns:
            Spearman correlation coefficient tensor
        """
        # Flatten inputs
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        if len(y_true_flat) < 2:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
            
        # Convert to ranks
        _, y_true_indices = torch.sort(y_true_flat, stable=True)
        _, y_pred_indices = torch.sort(y_pred_flat, stable=True)
        
        # Create rank tensors
        y_true_ranks = torch.empty_like(y_true_indices, dtype=y_true.dtype)
        y_pred_ranks = torch.empty_like(y_pred_indices, dtype=y_pred.dtype)
        
        y_true_ranks[y_true_indices] = torch.arange(len(y_true_indices), dtype=y_true.dtype, device=y_true.device) + 1
        y_pred_ranks[y_pred_indices] = torch.arange(len(y_pred_indices), dtype=y_pred.dtype, device=y_pred.device) + 1
        
        return self.calculate_ic(y_true_ranks, y_pred_ranks)

    def calculate_ir(self, y_true: torch.Tensor, y_pred: torch.Tensor, window: int = 30) -> torch.Tensor:
        """
        Calculate Information Ratio (IR) as Mean(Rolling IC) / Std(Rolling IC).
        
        Args:
            y_true: Actual prices tensor
            y_pred: Predicted prices tensor
            window: Rolling window size for IC calculation
            
        Returns:
            Information Ratio tensor
        """
        if y_true.numel() < 10 or y_pred.numel() < 10:
            return torch.tensor(0.0, dtype=y_true.dtype)

        # Use smaller window if sequence is short
        min_len = min(y_true.numel(), y_pred.numel())
        effective_window = min(window, max(5, min_len // 2))

        true_returns = self.calculate_returns(y_true)
        pred_returns = self.calculate_returns(y_pred)

        # Ensure returns are aligned
        min_len = min(len(true_returns), len(pred_returns))
        true_returns = true_returns[:min_len]
        pred_returns = pred_returns[:min_len]

        # Calculate rolling correlations more efficiently
        if len(true_returns) < effective_window:
            return torch.tensor(0.0, dtype=y_true.dtype)

        # Use unfold to get sliding windows
        true_windows = true_returns.unfold(0, effective_window, 1)
        pred_windows = pred_returns.unfold(0, effective_window, 1)

        # Calculate correlations for all windows at once
        rolling_ics = []
        for i in range(true_windows.size(0)):
            true_win = true_windows[i]
            pred_win = pred_windows[i]
            
            # Calculate correlation for this window
            combined = torch.stack([true_win, pred_win])
            corr_matrix = torch.corrcoef(combined)
            
            if not torch.isnan(corr_matrix[0, 1]):
                rolling_ics.append(corr_matrix[0, 1])
            else:
                rolling_ics.append(torch.tensor(0.0, dtype=y_true.dtype))

        if not rolling_ics:
            return torch.tensor(0.0, dtype=y_true.dtype)

        rolling_ics_tensor = torch.stack(rolling_ics)
        ic_mean = rolling_ics_tensor.mean()
        ic_std = rolling_ics_tensor.std(unbiased=False)  # Using biased std like in the original

        if ic_std == 0 or torch.isnan(ic_std):
            return torch.tensor(0.0, dtype=y_true.dtype)

        ir = ic_mean / ic_std
        return ir if not torch.isnan(ir) else torch.tensor(0.0, dtype=y_true.dtype)

    def calculate_max_deviation(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate the maximum absolute deviation between true and predicted values"""
        return torch.max(torch.abs(y_true - y_pred))

    def calculate_direction_accuracy(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate accuracy of directional movement prediction"""
        # Get differences between consecutive values
        true_diffs = torch.diff(y_true.flatten())
        pred_diffs = torch.diff(y_pred.flatten())
        
        if len(true_diffs) == 0:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
        
        # Count cases where both moved in the same direction
        same_direction = torch.sign(true_diffs) == torch.sign(pred_diffs)
        return torch.mean(same_direction.float())

    def calculate_shape_similarity(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate similarity of the shapes of two sequences using correlation"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        if len(y_true_flat) < 2:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
        
        # Normalize the sequences
        std_true = torch.std(y_true_flat)
        std_pred = torch.std(y_pred_flat)
        
        if std_true == 0 or std_pred == 0:
            return torch.tensor(float('nan'), dtype=y_true.dtype)
            
        y_true_norm = (y_true_flat - torch.mean(y_true_flat)) / std_true
        y_pred_norm = (y_pred_flat - torch.mean(y_pred_flat)) / std_pred
        
        # Calculate correlation as a measure of shape similarity
        n = len(y_true_norm)
        correlation = torch.sum(y_true_norm * y_pred_norm) / (n - 1)
        
        return correlation

    def calculate_tracking_error(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate tracking error (volatility of the difference between true and predicted)"""
        diff = y_true.flatten() - y_pred.flatten()
        return torch.std(diff)

    def evaluate_regression(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate regression performance with various metrics including new indicators.
        
        Args:
            y_true: Ground truth values tensor
            y_pred: Predicted values tensor
            feature_names: Names of features if evaluating multiple outputs
            
        Returns:
            Dictionary containing regression metrics
        """
        # Ensure inputs are 1D tensors
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Remove NaN values
        mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
        if not mask.any():
            return {}

        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return {}

        # Calculate basic regression metrics
        mae = self.calculate_mae(y_true_clean, y_pred_clean)
        mse = self.calculate_mse(y_true_clean, y_pred_clean)
        rmse = self.calculate_rmse(y_true_clean, y_pred_clean)
        mape = self.calculate_mape(y_true_clean, y_pred_clean)
        r2 = self.calculate_r2(y_true_clean, y_pred_clean)

        # Calculate IC metrics using original price series
        ic = self.calculate_ic(y_true, y_pred)
        rank_ic = self.calculate_rank_ic(y_true, y_pred)
        ir = self.calculate_ir(y_true, y_pred)

        # Calculate additional metrics
        max_deviation = self.calculate_max_deviation(y_true_clean, y_pred_clean)
        direction_accuracy = self.calculate_direction_accuracy(y_true, y_pred)
        shape_similarity = self.calculate_shape_similarity(y_true, y_pred)
        tracking_error = self.calculate_tracking_error(y_true_clean, y_pred_clean)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'IC': ic,
            'Rank IC': rank_ic,
            'IR': ir,
            'Max Deviation': max_deviation,
            'Direction Accuracy': direction_accuracy,
            'Shape Similarity': shape_similarity,
            'Tracking Error': tracking_error
        }

        return metrics

    def evaluate_all(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Evaluate all regression metrics.
        
        Args:
            y_true: Ground truth values tensor
            y_pred: Predicted values tensor
            feature_names: Names of features if applicable
            
        Returns:
            Dictionary containing regression metrics
        """
        regression_metrics = self.evaluate_regression(y_true, y_pred, feature_names)

        return {
            'Regression_Metrics': regression_metrics
        }

    def print_results(self, results: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Print formatted results"""
        print("="*50)
        print("TORCH MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\nREGRESSION METRICS:")
        print("-"*20)
        reg_metrics = results['Regression_Metrics']
        
        for metric, value in reg_metrics.items():
            if torch.is_tensor(value) and not torch.isnan(value):
                print(f"{metric:<16}: {value.item():.6f}")
            elif torch.is_tensor(value) and torch.isnan(value):
                print(f"{metric:<16}: NaN")
            else:
                print(f"{metric:<16}: {value}")


def evaluate_predictions(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    print_results: bool = True
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convenience function to evaluate predictions with regression metrics.
    
    Args:
        y_true: Ground truth values tensor
        y_pred: Predicted values tensor
        feature_names: Names of features if applicable
        print_results: Whether to print results to console
        
    Returns:
        Dictionary containing regression metrics
    """
    evaluator = TorchModelEvaluator()
    results = evaluator.evaluate_all(y_true, y_pred, feature_names)

    if print_results:
        evaluator.print_results(results)

    return results


def evaluate_multiple_assets(
    asset_predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    print_results: bool = True
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Evaluate predictions for multiple assets.
    
    Args:
        asset_predictions: Dictionary mapping asset names to (true, pred) tuples of tensors
        print_results: Whether to print results to console
        
    Returns:
        Dictionary containing metrics for each asset
    """
    evaluator = TorchModelEvaluator()
    all_results = {}

    for asset_name, (y_true, y_pred) in asset_predictions.items():
        results = evaluator.evaluate_all(y_true, y_pred)
        all_results[asset_name] = results

    if print_results:
        print("="*60)
        print("MULTI-ASSET TORCH EVALUATION RESULTS")
        print("="*60)

        for asset_name, results in all_results.items():
            print(f"\nAsset: {asset_name}")
            print("-"*30)
            evaluator.print_results(results)

        # Aggregate metrics across all assets
        if len(asset_predictions) > 1:
            all_y_true = torch.cat([pair[0].flatten() for pair in asset_predictions.values()])
            all_y_pred = torch.cat([pair[1].flatten() for pair in asset_predictions.values()])
            
            overall_results = evaluator.evaluate_all(all_y_true, all_y_pred)
            
            print("\n" + "="*60)
            print("AGGREGATED RESULTS ACROSS ALL ASSETS")
            print("="*60)
            evaluator.print_results(overall_results)

    return all_results