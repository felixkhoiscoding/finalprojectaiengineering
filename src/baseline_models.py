"""
Baseline Models Module

This module contains baseline forecasting models for time-series prediction.
These models serve as benchmarks for more complex models.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def naive_forecast(train, test, value_col='WPU101704'):
    """
    Naive Forecast: Use last observed value as prediction for all future periods
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    last_value = train[value_col].iloc[-1]
    predictions = np.full(len(test), last_value)
    
    return predictions


def seasonal_naive_forecast(train, test, value_col='WPU101704', seasonal_period=12):
    """
    Seasonal Naive Forecast: Use value from same season last year
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
    seasonal_period : int
        Seasonal period (12 for monthly data)
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    # Get last seasonal_period values from training data
    last_season = train[value_col].iloc[-seasonal_period:].values
    
    # Repeat pattern for forecast horizon
    n_periods = len(test)
    predictions = np.tile(last_season, (n_periods // seasonal_period) + 1)[:n_periods]
    
    return predictions


def moving_average_forecast(train, test, value_col='WPU101704', window=12):
    """
    Simple Moving Average Forecast
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
    window : int
        Window size for moving average
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    # Calculate moving average from last 'window' values
    last_values = train[value_col].iloc[-window:].values
    ma_value = np.mean(last_values)
    
    # Use same MA value for all predictions
    predictions = np.full(len(test), ma_value)
    
    return predictions


def simple_exponential_smoothing(train, test, value_col='WPU101704', smoothing_level=0.2):
    """
    Simple Exponential Smoothing (SES)
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
    smoothing_level : float
        Alpha parameter (0-1)
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    # Fit SES model
    model = ExponentialSmoothing(
        train[value_col],
        trend=None,
        seasonal=None
    )
    fitted_model = model.fit(smoothing_level=smoothing_level, optimized=False)
    
    # Forecast
    predictions = fitted_model.forecast(steps=len(test))
    
    return predictions.values


def holt_linear_trend(train, test, value_col='WPU101704'):
    """
    Holt's Linear Trend (Double Exponential Smoothing)
    
    Captures trend but no seasonality
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    # Fit Holt's model
    model = ExponentialSmoothing(
        train[value_col],
        trend='add',  # Additive trend
        seasonal=None
    )
    fitted_model = model.fit()
    
    # Forecast
    predictions = fitted_model.forecast(steps=len(test))
    
    return predictions.values


def holt_winters(train, test, value_col='WPU101704', seasonal_periods=12, 
                 trend='add', seasonal='add'):
    """
    Holt-Winters (Triple Exponential Smoothing)
    
    Captures BOTH trend and seasonality - Expected BEST baseline
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
    seasonal_periods : int
        Seasonal period (12 for monthly data)
    trend : str
        Type of trend component: 'add' or 'mul'
    seasonal : str
        Type of seasonal component: 'add' or 'mul'
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        train[value_col],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fitted_model = model.fit()
    
    # Forecast
    predictions = fitted_model.forecast(steps=len(test))
    
    return predictions.values


def calculate_metrics(y_true, y_pred):
    """
    Calculate forecasting evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing MAE, RMSE, MAPE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics


def evaluate_baseline_models(train, test, value_col='WPU101704'):
    """
    Evaluate all baseline models and return results
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    value_col : str
        Column name for target variable
        
    Returns:
    --------
    results : dict
        Dictionary with predictions and metrics for each model
    """
    results = {}
    y_true = test[value_col].values
    
    # 1. Naive Forecast
    print("Evaluating Naive Forecast...")
    pred_naive = naive_forecast(train, test, value_col)
    results['Naive'] = {
        'predictions': pred_naive,
        'metrics': calculate_metrics(y_true, pred_naive)
    }
    
    # 2. Seasonal Naive
    print("Evaluating Seasonal Naive...")
    pred_seasonal = seasonal_naive_forecast(train, test, value_col)
    results['Seasonal_Naive'] = {
        'predictions': pred_seasonal,
        'metrics': calculate_metrics(y_true, pred_seasonal)
    }
    
    # 3. Moving Average (12-month)
    print("Evaluating Moving Average (12-month)...")
    pred_ma = moving_average_forecast(train, test, value_col, window=12)
    results['MA_12'] = {
        'predictions': pred_ma,
        'metrics': calculate_metrics(y_true, pred_ma)
    }
    
    # 4. Simple Exponential Smoothing
    print("Evaluating Simple Exponential Smoothing...")
    try:
        pred_ses = simple_exponential_smoothing(train, test, value_col)
        results['SES'] = {
            'predictions': pred_ses,
            'metrics': calculate_metrics(y_true, pred_ses)
        }
    except Exception as e:
        print(f"  [WARNING] SES failed: {e}")
        results['SES'] = {'predictions': None, 'metrics': None}
    
    # 5. Holt's Linear Trend
    print("Evaluating Holt's Linear Trend...")
    try:
        pred_holt = holt_linear_trend(train, test, value_col)
        results['Holt'] = {
            'predictions': pred_holt,
            'metrics': calculate_metrics(y_true, pred_holt)
        }
    except Exception as e:
        print(f"  [WARNING] Holt failed: {e}")
        results['Holt'] = {'predictions': None, 'metrics': None}
    
    # 6. Holt-Winters (Additive)
    print("Evaluating Holt-Winters (Additive)...")
    try:
        pred_hw = holt_winters(train, test, value_col, trend='add', seasonal='add')
        results['Holt_Winters'] = {
            'predictions': pred_hw,
            'metrics': calculate_metrics(y_true, pred_hw)
        }
    except Exception as e:
        print(f"  [WARNING] Holt-Winters failed: {e}")
        results['Holt_Winters'] = {'predictions': None, 'metrics': None}
    
    return results


def create_comparison_table(results):
    """
    Create comparison table of model performance
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_baseline_models()
        
    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by RMSE
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        if model_results['metrics'] is not None:
            row = {
                'Model': model_name,
                **model_results['metrics']
            }
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by RMSE (lower is better)
    comparison_df = comparison_df.sort_values('RMSE')
    
    # Add rank
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    
    return comparison_df


if __name__ == "__main__":
    print("Baseline models module loaded successfully!")
    print("\nAvailable functions:")
    print("  - naive_forecast()")
    print("  - seasonal_naive_forecast()")
    print("  - moving_average_forecast()")
    print("  - simple_exponential_smoothing()")
    print("  - holt_linear_trend()")
    print("  - holt_winters()")
    print("  - calculate_metrics()")
    print("  - evaluate_baseline_models()")
    print("  - create_comparison_table()")
