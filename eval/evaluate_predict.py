"""
Script to evaluate a Kronos model with test dataset and generate a comprehensive evaluation report.
The script computes both regression and classification metrics as specified in the requirements.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import Kronos, KronosTokenizer, KronosPredictor
from eval.evaluate import ModelEvaluator, evaluate_predictions
import warnings

warnings.filterwarnings("ignore")
# cd d:\dev\Kronos\eval; python d:\dev\Kronos\eval\evaluate_predict.py --model-path d:\dev\Kronos\pretrained\Kronos-small --tokenizer-path d:\dev\Kronos\pretrained\Kronos-Tokenizer-2k --test-data d:\dev\Kronos\download\data\000001_5min_20260217.csv --lookback 512 --pred-len 48 --output-file rep.txt --plot-output rep.png


def load_test_data(test_data_path):
    """
    Load test dataset from file.

    Args:
        test_data_path (str): Path to the test data CSV file

    Returns:
        pd.DataFrame: Loaded test data
    """
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    df = pd.read_csv(test_data_path)

    # Check if required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns in test data: {missing_cols}")

    if "timestamps" not in df.columns and "date" in df.columns:
        # If no timestamps, create them assuming daily intervals
        df["timestamps"] = pd.to_datetime(df["date"], utc=True)
    else:
        # If no timestamp column exists, create one
        if "timestamps" not in df.columns:
            raise ValueError(
                "No 'timestamps' column found in test data. Please provide a 'timestamps' column or specify a 'date' column."
            )
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    return df


def plot_prediction(
    actual_values, predicted_values, metrics, output_path=None, historical_values=None
):
    """
    Generate a plot comparing actual and predicted values.
    Red line: Prediction
    Blue line: Actual
    Green line: History (optional)
    Write metrics on the plot.
    """
    plt.figure(figsize=(12, 6))

    start_idx = 0
    if historical_values is not None:
        hist_len = len(historical_values)
        # Plot history
        plt.plot(
            range(hist_len),
            historical_values,
            label="History",
            color="green",
            linewidth=1.5,
        )

        # Add a vertical line to separate history and future
        plt.axvline(x=hist_len - 1, color="gray", linestyle=":", alpha=0.5)
        start_idx = hist_len

    # Plot actual and predicted lines
    indices = range(start_idx, start_idx + len(actual_values))
    plt.plot(indices, actual_values, label="True Future", color="blue", linewidth=2)
    plt.plot(
        indices, predicted_values, label="Predicted Future", color="red", linewidth=2
    )

    # Create text string for metrics
    text_str = "\n".join(
        [f"{k}: {v:.4f}" for k, v in metrics.items() if not np.isnan(v)]
    )

    # Add text box
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.gca().text(
        0.05,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.title("Stock Price Prediction vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if output_path:
        # Create directory if not exists
        (
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.dirname(output_path)
            else None
        )
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        # If running interactively, show plot
        try:
            plt.show()
        except:
            pass
    plt.close()


def prepare_sequences(df, lookback, pred_len):
    """
    Prepare input-output sequences from the test data.
    The sequences are divided by day. For each day, we use the previous `lookback` points
    to predict the `pred_len` points of that day (open to close).

    Args:
        df (pd.DataFrame): Input dataframe with financial data
        lookback (int): Number of past timesteps to use as input
        pred_len (int): Number of future timesteps to predict

    Returns:
        list: List of (input_sequence, output_sequence) pairs
    """
    sequences = []
    
    # Ensure timestamps are datetime objects
    # Use pandas check which handles timezone-aware datetimes correctly
    if not pd.api.types.is_datetime64_any_dtype(df['timestamps']):
         df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True)
         
    # Group data by date
    # Create a column for date to group by
    df['date_group'] = df['timestamps'].dt.date
    unique_dates = df['date_group'].unique()
    
    # Sort dates to ensure chronological order
    unique_dates = sorted(unique_dates)
    
    print(f"Found {len(unique_dates)} unique dates in dataset.")
    
    # Iterate through each day starting from the second day (need history)
    # We can't predict the first day if we need history from previous days,
    # unless history is contained within the same day (intra-day), 
    # but the requirement says "previous 512 time points", which likely span multiple days.
    
    # Sort dates to ensure chronological order
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # Re-create date_group after sort/reset to be safe although it should persist
    df['date_group'] = df['timestamps'].dt.date
    unique_dates = sorted(df['date_group'].unique())

    print(f"Found {len(unique_dates)} unique dates in dataset.")

    for date_val in unique_dates:
        # Get data for this day using boolean mask on the sorted dataframe
        day_indices = df.index[df['date_group'] == date_val].tolist()
        
        if not day_indices:
            continue
            
        start_idx = day_indices[0]
        end_idx = day_indices[-1]
        day_len = end_idx - start_idx + 1
        
        if day_len < pred_len:
             # Skip days that don't have enough points
             continue
        
        # Check if we have enough history
        # We need lookback points BEFORE the start of the day
        if start_idx < lookback:
            continue
            
        # Extract input and output sequences
        # Input: The lookback points immediately preceding the day
        input_seq = df.iloc[start_idx - lookback : start_idx].copy()
        
        # Output: The first pred_len points of the day
        output_seq = df.iloc[start_idx : start_idx + pred_len].copy() 
        
        sequences.append((input_seq, output_seq))

    return sequences


def evaluate_model(
    model_path,
    tokenizer_path,
    test_data_path,
    lookback=512,
    pred_len=48,
    output_file=None,
    plot_output_path=None,
):
    """
    Evaluate the model with the specified test data and generate a report.

    Args:
        model_path (str): Path to the pretrained model
        tokenizer_path (str): Path to the pretrained tokenizer
        test_data_path (str): Path to the test data CSV file
        lookback (int): Number of past timesteps to use as input
        pred_len (int): Number of future timesteps to predict
        output_file (str, optional): Path to save the evaluation report
        plot_output_path (str, optional): Path to save the prediction plot
    """
    print("Loading model and tokenizer...")

    # Load model and tokenizer
    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        print(f"Successfully loaded model from: {model_path}")
        print(f"Successfully loaded tokenizer from: {tokenizer_path}")
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_df = load_test_data(test_data_path)
    print(f"Test data shape: {test_df.shape}")

    # Prepare sequences
    print(f"Preparing sequences with lookback={lookback}, pred_len={pred_len}")
    sequences = prepare_sequences(test_df, lookback, pred_len)
    print(f"Number of sequences prepared: {len(sequences)}")

    if len(sequences) == 0:
        print(
            "Not enough data to create sequences. Either increase the dataset size or decrease lookback/pred_len."
        )
        return

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Metrics dictionaries to accumulate results across all sequences
    # Use defaultdict to handle any new metrics that might be added
    from collections import defaultdict

    regression_metrics_all = defaultdict(list)
    classification_metrics_all = defaultdict(list)

    print("Starting model evaluation...")

    # Process each sequence
    # evaluate all valid sequences (days)
    total_sequences = len(sequences)
    
    for idx, (input_seq, output_seq) in enumerate(sequences):
        print(f"Evaluating sequence {idx+1}/{total_sequences}")

        try:
            # Prepare input for prediction
            x_df = (
                input_seq[["open", "high", "low", "close", "volume", "amount"]].fillna(
                    0
                )
                if "amount" in input_seq.columns
                else input_seq[["open", "high", "low", "close", "volume"]].assign(
                    amount=lambda x: x["volume"] * x["close"]
                )
            )
            x_timestamp = input_seq["timestamps"]
            y_timestamp = output_seq["timestamps"]

            # Make prediction
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=0.1,
                top_k=0,
                top_p=0.9,
                sample_count=5,
                verbose=True,
            )

            # Extract actual values
            actual_values = output_seq[
                ["open", "high", "low", "close", "volume"]
            ].values
            predicted_values = pred_df[
                ["open", "high", "low", "close", "volume"]
            ].values
            input_values = input_seq[["open", "high", "low", "close", "volume"]].values

            # Flatten arrays to treat each value independently

            actual_flat = [
                round(
                    (
                        actual_values[i][0]
                        + actual_values[i][1]
                        + actual_values[i][2]
                        + actual_values[i][3]
                    )
                    / 4,
                    2,
                )
                for i in range(len(actual_values))
            ]
            pred_flat = [
                round(
                    (
                        predicted_values[i][0]
                        + predicted_values[i][1]
                        + predicted_values[i][2]
                        + predicted_values[i][3]
                    )
                    / 4,
                    2,
                )
                for i in range(len(predicted_values))
            ]
            input_flat = [
                round(
                    (
                        input_values[i][0]
                        + input_values[i][1]
                        + input_values[i][2]
                        + input_values[i][3]
                    )
                    / 4,
                    2,
                )
                for i in range(len(input_values))
            ]

            # Compute metrics for this sequence
            seq_reg_metrics = evaluator.evaluate_regression(actual_flat, pred_flat)
            seq_clf_metrics = evaluator.evaluate_classification(actual_flat, pred_flat)

            # Accumulate metrics
            for metric, value in seq_reg_metrics.items():
                if not np.isnan(value):
                    regression_metrics_all[metric].append(value)

            for metric, value in seq_clf_metrics.items():
                if not np.isnan(value):
                    classification_metrics_all[metric].append(value)

            print(f"Sequence {idx+1} metrics:")
            print(f"Regression metrics: {seq_reg_metrics}")
            print(f"Classification metrics: {seq_clf_metrics}")
            # Save plot for the last sequence (as a representative sample)
            # if plot_output_path and idx == min(len(sequences), 20) - 1:
            # Format metrics for display
            display_metrics = {}
            for k, v in seq_reg_metrics.items():
                if k in ["MAE", "RMSE", "IC", "IR", "R²"] and not np.isnan(v):
                    display_metrics[k] = v
            plot_prediction(
                actual_flat,
                pred_flat,
                display_metrics,
                plot_output_path,
                historical_values=input_flat,
            )
            print(f"Prediction plot saved to: {plot_output_path}")

        except Exception as e:
            print(f"Error processing sequence {idx+1}: {e}")
            continue

    print("Computing final metrics...")

    # Compute mean metrics across all sequences
    final_regression_metrics = {}
    for metric, values in regression_metrics_all.items():
        if values:
            final_regression_metrics[metric] = np.mean(values)
        else:
            final_regression_metrics[metric] = float("nan")

    final_classification_metrics = {}
    for metric, values in classification_metrics_all.items():
        if values:
            final_classification_metrics[metric] = np.mean(values)
        else:
            final_classification_metrics[metric] = float("nan")

    # Generate report
    report = generate_report(
        final_regression_metrics,
        final_classification_metrics,
        model_path,
        test_data_path,
        lookback,
        pred_len,
        len(sequences),
    )

    # Print report to console
    print(report)

    # Save report to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")


def generate_report(
    regression_metrics,
    classification_metrics,
    model_path,
    test_data_path,
    lookback,
    pred_len,
    total_sequences,
):
    """
    Generate a formatted evaluation report.

    Args:
        regression_metrics (dict): Regression metrics
        classification_metrics (dict): Classification metrics
        model_path (str): Path to the evaluated model
        test_data_path (str): Path to the test data
        lookback (int): Lookback length used
        pred_len (int): Prediction length used
        total_sequences (int): Total number of sequences processed

    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 70)
    report.append("KRONOS MODEL EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("CONFIGURATION")
    report.append("-" * 20)
    report.append(f"Model: {model_path}")
    report.append(f"Test Data: {test_data_path}")
    report.append(f"Lookback Length: {lookback}")
    report.append(f"Prediction Length: {pred_len}")
    report.append(f"Total Sequences Evaluated: {total_sequences}")
    report.append("")

    report.append("REGRESSION METRICS (Continuous Value Prediction)")
    report.append("-" * 50)
    report.append(
        "These metrics evaluate the model's ability to predict continuous values like stock prices and volumes."
    )
    report.append("")
    report.append(
        "MAE (Mean Absolute Error): Average magnitude of errors without considering direction"
    )
    report.append(f"  Value: {regression_metrics.get('MAE', float('nan')):.6f}")
    report.append(
        f"  Interpretation: On average, predictions deviate by {regression_metrics.get('MAE', float('nan')):.6f} units"
    )
    report.append("")

    report.append(
        "MSE (Mean Squared Error): Penalizes larger errors more heavily than smaller ones"
    )
    report.append(f"  Value: {regression_metrics.get('MSE', float('nan')):.6f}")
    report.append(
        "  Interpretation: Larger errors are penalized quadratically, indicating extreme deviations"
    )
    report.append("")

    report.append(
        "RMSE (Root Mean Squared Error): Square root of MSE, in the same units as target"
    )
    report.append(f"  Value: {regression_metrics.get('RMSE', float('nan')):.6f}")
    report.append(
        "  Interpretation: Similar to MSE but in original units, showing severity of errors"
    )
    report.append("")

    report.append("MAPE (Mean Absolute Percentage Error): Relative error as percentage")
    report.append(f"  Value: {regression_metrics.get('MAPE', float('nan')):.6f}")
    report.append(
        "  Interpretation: Relative error, especially useful for different price scales"
    )
    report.append("")

    report.append(
        "R² (Coefficient of Determination): Proportion of variance explained by model"
    )
    report.append(f"  Value: {regression_metrics.get('R²', float('nan')):.6f}")
    report.append(
        "  Interpretation: How well the model explains the variability of the target"
    )
    report.append("")

    report.append("IC (Information Coefficient): Pearson correlation of returns")
    report.append(f"  Value: {regression_metrics.get('IC', float('nan')):.6f}")
    report.append(
        "  Interpretation: Linear correlation between predicted and actual returns"
    )
    report.append("")

    report.append(
        "Rank IC (Rank Information Coefficient): Spearman correlation of returns"
    )
    report.append(f"  Value: {regression_metrics.get('Rank IC', float('nan')):.6f}")
    report.append("  Interpretation: Rank correlation, robust to outliers")
    report.append("")

    report.append("IR (Information Ratio): Mean(IC) / Std(IC)")
    report.append(f"  Value: {regression_metrics.get('IR', float('nan')):.6f}")
    report.append(
        "  Interpretation: Stability of the prediction signal (Rolling IC based)"
    )
    report.append("")

    report.append("CLASSIFICATION METRICS (Direction Prediction)")
    report.append("-" * 50)
    report.append(
        "These metrics evaluate the model's ability to predict price direction (up/down)."
    )
    report.append("")

    report.append("Accuracy: Overall proportion of correct direction predictions")
    report.append(
        f"  Value: {classification_metrics.get('Accuracy', float('nan')):.6f}"
    )
    report.append(
        f"  Interpretation: {(classification_metrics.get('Accuracy', float('nan'))*100):.2f}% of direction predictions were correct"
    )
    report.append("")

    report.append(
        "Precision: Proportion of positive identifications that were actually correct"
    )
    report.append(
        f"  Value: {classification_metrics.get('Precision', float('nan')):.6f}"
    )
    report.append(
        "  Interpretation: Of all upward moves predicted, this fraction was actually up"
    )
    report.append("")

    report.append("Recall: Proportion of actual positives identified correctly")
    report.append(f"  Value: {classification_metrics.get('Recall', float('nan')):.6f}")
    report.append(
        "  Interpretation: Of all actual upward moves, this fraction was caught"
    )
    report.append("")

    report.append("F1-Score: Harmonic mean of precision and recall")
    report.append(
        f"  Value: {classification_metrics.get('F1-Score', float('nan')):.6f}"
    )
    report.append("  Interpretation: Balance between precision and recall")
    report.append("")

    report.append("AUC-ROC: Area Under the ROC Curve")
    report.append(f"  Value: {classification_metrics.get('AUC-ROC', float('nan')):.6f}")
    report.append("  Interpretation: Ability to distinguish between up/down movements")
    report.append("")

    report.append("Kappa Coefficient: Agreement adjusted for chance")
    report.append(f"  Value: {classification_metrics.get('Kappa', float('nan')):.6f}")
    report.append("  Interpretation: Performance compared to random guessing")
    report.append("")

    report.append("=" * 70)
    report.append("REPORT GENERATED")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Kronos model with test dataset"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the pretrained tokenizer",
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=512,
        help="Number of past timesteps to use as input (default: 512)",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=48,
        help="Number of future timesteps to predict (default: 48)",
    )
    parser.add_argument(
        "--output-file", type=str, default="req.txt", help="Path to save the evaluation report (optional)"
    )
    parser.add_argument(
        "--plot-output", type=str, default="plot.png", help="Path to save the prediction plot (optional)"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        test_data_path=args.test_data,
        lookback=args.lookback,
        pred_len=args.pred_len,
        output_file=args.output_file,
        plot_output_path=args.plot_output,
    )


if __name__ == "__main__":
    main()
