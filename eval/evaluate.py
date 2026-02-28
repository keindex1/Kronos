import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A comprehensive evaluator for financial forecasting models that computes both
    regression and classification metrics for stock price prediction.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """
        Calculate log returns from prices.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of log returns
        """
        return np.diff(np.log(prices), prepend=np.nan)[1:]
    
    @staticmethod
    def classify_direction(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous predictions to binary direction (up/down) classification.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Tuple of actual and predicted directions (1 for up, 0 for down)
        """
        # Calculate returns
        actual_returns = np.diff(actual, prepend=actual[0])
        predicted_returns = np.diff(predicted, prepend=predicted[0])
        
        # Ensure alignment
        min_len = min(len(actual_returns), len(predicted_returns))
        actual_returns = actual_returns[:min_len]
        predicted_returns = predicted_returns[:min_len]
        
        # Create common mask for valid values in both
        mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
        
        if np.sum(mask) == 0:
            print("WARNING: classify_direction - all diffs are NaN")
            print(f"actual_returns sample: {actual_returns[:5]}")
            print(f"predicted_returns sample: {predicted_returns[:5]}")
        
        # Calculate directions only for valid, aligned data points
        actual_direction = (actual_returns[mask] > 0).astype(int)
        predicted_direction = (predicted_returns[mask] > 0).astype(int)
        
        return actual_direction, predicted_direction
    
    def calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Information Coefficient (IC) - Pearson correlation between actual and predicted returns.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            
        Returns:
            Pearson correlation coefficient
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
            
        true_returns = self.calculate_returns(y_true)
        pred_returns = self.calculate_returns(y_pred)
        
        # Ensure returns are aligned
        min_len = min(len(true_returns), len(pred_returns))
        true_returns = true_returns[:min_len]
        pred_returns = pred_returns[:min_len]
        
        # Filter NaNs
        mask = ~(np.isnan(true_returns) | np.isnan(pred_returns))
        if np.sum(mask) < 2:
            return 0.0
            
        s1 = pd.Series(true_returns[mask])
        s2 = pd.Series(pred_returns[mask])
        r = s1.corr(s2, method='pearson')
        return r if not np.isnan(r) else 0.0

    def calculate_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Rank Information Coefficient (Rank IC) - Spearman correlation between actual and predicted returns.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            
        Returns:
            Spearman correlation coefficient
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
            
        true_returns = self.calculate_returns(y_true)
        pred_returns = self.calculate_returns(y_pred)
        
        # Ensure returns are aligned
        min_len = min(len(true_returns), len(pred_returns))
        true_returns = true_returns[:min_len]
        pred_returns = pred_returns[:min_len]
        
        # Filter NaNs
        mask = ~(np.isnan(true_returns) | np.isnan(pred_returns))
        if np.sum(mask) < 2:
            return 0.0
            
        s1 = pd.Series(true_returns[mask])
        s2 = pd.Series(pred_returns[mask])
        r = s1.corr(s2, method='spearman')
        return r if not np.isnan(r) else 0.0

    def calculate_ir(self, y_true: np.ndarray, y_pred: np.ndarray, window: int = 30) -> float:
        """
        Calculate Information Ratio (IR) as Mean(Rolling IC) / Std(Rolling IC).
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            window: Rolling window size for IC calculation
            
        Returns:
            Information Ratio
        """
        if len(y_true) < 10 or len(y_pred) < 10:
            return 0.0
            
        # Use smaller window if sequence is short
        min_len = min(len(y_true), len(y_pred))
        effective_window = min(window, max(5, min_len // 2))
        
        true_returns = self.calculate_returns(y_true)
        pred_returns = self.calculate_returns(y_pred)
        
        # Ensure returns are aligned
        min_len = min(len(true_returns), len(pred_returns))
        true_returns = true_returns[:min_len]
        pred_returns = pred_returns[:min_len]
        
        # Filter NaNs
        mask = ~(np.isnan(true_returns) | np.isnan(pred_returns))
        if np.sum(mask) < effective_window + 2:
            return 0.0
            
        s1 = pd.Series(true_returns[mask])
        s2 = pd.Series(pred_returns[mask])
        
        rolling_ic = s1.rolling(window=effective_window).corr(s2)
        
        # Drop NaNs from rolling calc
        rolling_ic = rolling_ic.dropna()
        
        if len(rolling_ic) == 0:
            return 0.0

        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        
        if ic_std == 0 or np.isnan(ic_std):
            return 0.0
            
        ir = ic_mean / ic_std
        return ir if not np.isnan(ir) else 0.0

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate regression performance with various metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            feature_names: Names of features if evaluating multiple outputs
            
        Returns:
            Dictionary containing regression metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE calculation with handling for zero values
        mape = mean_absolute_percentage_error(y_true_clean[y_true_clean != 0], 
                                              y_pred_clean[y_true_clean != 0])
        
        # Calculate IC metrics
        # Note: We use the original arrays (before cleaning) to preserve time order for return calculation
        ic = self.calculate_ic(y_true, y_pred)
        rank_ic = self.calculate_rank_ic(y_true, y_pred)
        ir = self.calculate_ir(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'IC': ic,
            'Rank IC': rank_ic,
            'IR': ir
        }
        
        return metrics
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average_method: str = 'binary'
    ) -> Dict[str, float]:
        """
        Evaluate classification performance for direction prediction.
        
        Args:
            y_true: Ground truth values (continuous)
            y_pred: Predicted values (continuous)
            average_method: Method for multiclass averaging ('binary', 'macro', 'micro', 'weighted')
            
        Returns:
            Dictionary containing classification metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Convert to direction classification
        actual_direction, predicted_direction = self.classify_direction(y_true, y_pred)
        
        if len(actual_direction) == 0 or len(predicted_direction) == 0:
            print("WARNING: Classification Failed - Empty direction arrays")
            print(f"y_true sample: {y_true[:5]}")
            print(f"y_pred sample: {y_pred[:5]}")
            return {}
        
        # Calculate classification metrics
        accuracy = accuracy_score(actual_direction, predicted_direction)
        
        # Handle cases where precision/recall/f1 can't be calculated due to lack of positive samples
        try:
            precision = precision_score(actual_direction, predicted_direction, average=average_method, zero_division=0)
        except:
            precision = 0
        
        try:
            recall = recall_score(actual_direction, predicted_direction, average=average_method, zero_division=0)
        except:
            recall = 0
            
        try:
            f1 = f1_score(actual_direction, predicted_direction, average=average_method, zero_division=0)
        except:
            f1 = 0
            
        try:
            auc = roc_auc_score(actual_direction, predicted_direction)
        except:
            auc = 0.5  # Random classifier AUC if calculation fails
            
        try:
            kappa = cohen_kappa_score(actual_direction, predicted_direction)
        except:
            kappa = 0
            
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'Kappa': kappa
        }
        
        return metrics
    
    def evaluate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both regression and classification metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            feature_names: Names of features if applicable
            
        Returns:
            Dictionary containing both regression and classification metrics
        """
        regression_metrics = self.evaluate_regression(y_true, y_pred, feature_names)
        classification_metrics = self.evaluate_classification(y_true, y_pred)
        
        return {
            'Regression_Metrics': regression_metrics,
            'Classification_Metrics': classification_metrics
        }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[List[str]] = None,
    print_results: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to evaluate predictions with all metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        feature_names: Names of features if applicable
        print_results: Whether to print results to console
        
    Returns:
        Dictionary containing both regression and classification metrics
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(y_true, y_pred, feature_names)
    
    if print_results:
        print("="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\nREGRESSION METRICS:")
        print("-"*20)
        for metric, value in results['Regression_Metrics'].items():
            print(f"{metric:<12}: {value:.6f}")
        
        print("\nCLASSIFICATION METRICS (Direction Prediction):")
        print("-"*45)
        for metric, value in results['Classification_Metrics'].items():
            print(f"{metric:<15}: {value:.6f}")
    
    return results


def evaluate_multiple_assets(
    asset_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    print_results: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate predictions for multiple assets.
    
    Args:
        asset_predictions: Dictionary mapping asset names to (true, pred) tuples
        print_results: Whether to print results to console
        
    Returns:
        Dictionary containing metrics for each asset
    """
    all_results = {}
    
    for asset_name, (y_true, y_pred) in asset_predictions.items():
        results = evaluate_predictions(y_true, y_pred, print_results=False)
        all_results[asset_name] = results
    
    if print_results:
        print("="*60)
        print("MULTI-ASSET EVALUATION RESULTS")
        print("="*60)
        
        for asset_name, results in all_results.items():
            print(f"\nAsset: {asset_name}")
            print("-"*30)
            
            print("Regression Metrics:")
            for metric, value in results['Regression_Metrics'].items():
                print(f"  {metric:<12}: {value:.6f}")
                
            print("Classification Metrics:")
            for metric, value in results['Classification_Metrics'].items():
                print(f"  {metric:<15}: {value:.6f}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Script - Testing with Sample Data")
    print("="*55)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate stock prices with some trend and noise
    returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting price of 100
    
    # Create predictions with some correlation to actual
    noise = np.random.normal(0, 0.01, n_samples)
    predicted_prices = prices + noise
    
    # Run evaluation
    results = evaluate_predictions(
        y_true=prices,
        y_pred=predicted_prices,
        print_results=True
    )
    
    # Example with multiple assets
    print("\n\n" + "="*60)
    print("MULTIPLE ASSETS EXAMPLE")
    print("="*60)
    
    multiple_assets = {
        'AAPL': (prices, predicted_prices),
        'GOOGL': (prices * 1.2, predicted_prices * 1.21),  # Different scale
        'TSLA': (prices * 0.8, predicted_prices * 0.79)   # Different scale
    }
    
    multi_results = evaluate_multiple_assets(multiple_assets, print_results=True)