"""
Evaluation Module

Comprehensive model comparison and evaluation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, model_name='Model'):
    """
    Calculate comprehensive forecasting metrics
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict : Metrics dictionary
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def compare_all_models(train_df, test_df):
    """
    Evaluate all 9 models and return comparison
    
    Returns:
    --------
    pd.DataFrame : Comparison table sorted by RMSE
    """
    from src.baseline_models import (
        naive_forecast, seasonal_naive_forecast, moving_average_forecast,
        simple_exponential_smoothing, holt_linear_trend, holt_winters
    )
    from src.advanced_models import train_arima, train_prophet, train_lstm
    import joblib
    from pathlib import Path
    
    results = []
    y_true = test_df['WPU101704'].values
    
    # 1. Naive
    pred = naive_forecast(train_df, test_df, 'WPU101704')
    results.append(calculate_metrics(y_true, pred, 'Naive'))
    
    # 2. Seasonal Naive
    pred = seasonal_naive_forecast(train_df, test_df, 'WPU101704')
    results.append(calculate_metrics(y_true, pred, 'Seasonal Naive'))
    
    # 3. MA-12
    pred = moving_average_forecast(train_df, test_df, 'WPU101704', window=12)
    results.append(calculate_metrics(y_true, pred, 'MA-12'))
    
    # 4. SES
    try:
        pred = simple_exponential_smoothing(train_df, test_df, 'WPU101704')
        results.append(calculate_metrics(y_true, pred, 'SES'))
    except:
        pass
    
    # 5. Holt's
    try:
        pred = holt_linear_trend(train_df, test_df, 'WPU101704')
        results.append(calculate_metrics(y_true, pred, "Holt's"))
    except:
        pass
    
    # 6. Holt-Winters
    try:
        pred = holt_winters(train_df, test_df, 'WPU101704')
        results.append(calculate_metrics(y_true, pred, 'Holt-Winters'))
    except:
        pass
    
    # 7. ARIMA
    try:
        pred, _ = train_arima(train_df, test_df, order=(1,1,1), seasonal=False)
        results.append(calculate_metrics(y_true, pred, 'ARIMA'))
    except:
        pass
    
    # 8. Prophet
    try:
        pred, _ = train_prophet(train_df, test_df)
        results.append(calculate_metrics(y_true, pred, 'Prophet'))
    except:
        pass
    
    # 9. LSTM
    try:
        project_root = Path(__file__).parent.parent
        processed_dir = project_root / 'data' / 'processed'
        X_train = np.load(processed_dir / 'X_train_lstm.npy')
        y_train = np.load(processed_dir / 'y_train_lstm.npy')
        X_test = np.load(processed_dir / 'X_test_lstm.npy')
        scaler = joblib.load(processed_dir / 'lstm_scaler.pkl')
        
        pred_scaled, _, _ = train_lstm(X_train, y_train, X_test, epochs=100, batch_size=16)
        pred = scaler.inverse_transform(pred_scaled).flatten()
        results.append(calculate_metrics(y_true, pred, 'LSTM'))
    except:
        pass
    
    # Create dataframe and sort
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('RMSE').reset_index(drop=True)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    
    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """
    Create comparison visualization
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE comparison
    axes[0].barh(comparison_df['Model'], comparison_df['RMSE'], 
                 color=['green' if i == 0 else 'steelblue' for i in range(len(comparison_df))])
    axes[0].set_xlabel('RMSE (Lower is Better)', fontsize=12)
    axes[0].set_title('Model Performance: RMSE', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # MAPE comparison
    axes[1].barh(comparison_df['Model'], comparison_df['MAPE'],
                 color=['green' if i == 0 else 'coral' for i in range(len(comparison_df))])
    axes[1].set_xlabel('MAPE % (Lower is Better)', fontsize=12)
    axes[1].set_title('Model Performance: MAPE', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_predictions(train_df, test_df, predictions_dict, save_path=None):
    """
    Plot actual vs predicted for all models
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    predictions_dict : dict
        Dictionary of {model_name: predictions}
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot training data
    ax.plot(train_df['observation_date'], train_df['WPU101704'],
            label='Training Data', color='gray', alpha=0.5, linewidth=1)
    
    # Plot test data (actual)
    ax.plot(test_df['observation_date'], test_df['WPU101704'],
            label='Actual (Test)', color='black', linewidth=2.5, marker='o', markersize=6)
    
    # Plot predictions
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        ax.plot(test_df['observation_date'], preds,
                label=model_name, color=colors[i % len(colors)], 
                linewidth=2, marker='x', markersize=8, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('PPI Value', fontsize=13)
    ax.set_title('Model Predictions Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Evaluation module loaded!")
    print("Available functions:")
    print("  - calculate_metrics()")
    print("  - compare_all_models()")
    print("  - plot_model_comparison()")
    print("  - plot_predictions()")
