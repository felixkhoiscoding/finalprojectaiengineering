"""
Data Preprocessing Module

This module contains functions for preparing time-series data for modeling.
Includes train/test split, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import TEST_SIZE, RANDOM_SEED


def train_test_split_ts(data, date_col='observation_date', test_size=None):
    """
    Split time series data into train and test sets
    
    IMPORTANT: Uses temporal ordering - test set is the LAST n observations
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series dataframe
    date_col : str
        Name of the date column
    test_size : int, optional
        Number of observations for test set (default from config)
        
    Returns:
    --------
    train_df, test_df : tuple of pd.DataFrame
        Training and testing datasets
    """
    # USER INPUT: test_size from config if not provided
    if test_size is None:
        test_size = TEST_SIZE
    
    # Sort by date to ensure temporal order
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    
    # Split
    split_idx = len(data_sorted) - test_size
    train_df = data_sorted.iloc[:split_idx].copy()
    test_df = data_sorted.iloc[split_idx:].copy()
    
    print(f"Train/Test Split:")
    print(f"  Total observations: {len(data_sorted)}")
    print(f"  Train size: {len(train_df)} ({len(train_df)/len(data_sorted)*100:.1f}%)")
    print(f"  Test size: {len(test_df)} ({len(test_df)/len(data_sorted)*100:.1f}%)")
    print(f"  Train period: {train_df[date_col].min()} to {train_df[date_col].max()}")
    print(f"  Test period: {test_df[date_col].min()} to {test_df[date_col].max()}")
    
    return train_df, test_df


def create_lag_features(data, value_col='WPU101704', lags=[1, 3, 6, 12]):
    """
    Create lag features for time series
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to create lags from
    lags : list
        List of lag periods to create
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag columns
    """
    df = data.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
    
    print(f"Created {len(lags)} lag features: {lags}")
    print(f"  Columns added: {[f'{value_col}_lag_{lag}' for lag in lags]}")
    print(f"  Note: First {max(lags)} rows will have NaN in lag features")
    
    return df


def create_rolling_features(data, value_col='WPU101704', windows=[3, 6, 12]):
    """
    Create rolling statistics features
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to calculate rolling stats from
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added rolling features
    """
    df = data.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window=window).mean()
        
        # Rolling std
        df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window=window).std()
    
    print(f"Created rolling features for windows: {windows}")
    print(f"  Features per window: mean, std")
    print(f"  Total features added: {len(windows) * 2}")
    
    return df


def create_difference_features(data, value_col='WPU101704', periods=[1, 12]):
    """
    Create differenced features
    
    Based on Step 2 findings: Series is non-stationary, requires differencing
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to difference
    periods : list
        List of difference periods (1=MoM, 12=YoY for monthly data)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added difference columns
    """
    df = data.copy()
    
    for period in periods:
        df[f'{value_col}_diff_{period}'] = df[value_col].diff(periods=period)
    
    print(f"Created difference features for periods: {periods}")
    print(f"  diff_1: Month-over-month change")
    print(f"  diff_12: Year-over-year change")
    
    return df


def create_time_features(data, date_col='observation_date'):
    """
    Create time-based features from date column
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series dataframe
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
    """
    df = data.copy()
    
    # Extract time components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    
    # Cyclical encoding for month (useful for LSTM)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"Created time features:")
    print(f"  - year, month, quarter")
    print(f"  - month_sin, month_cos (cyclical encoding)")
    
    return df


def scale_data(train_data, test_data=None, columns=None, scaler_type='minmax'):
    """
    Scale data using MinMax or Standard scaler
    
    IMPORTANT: Fit scaler on TRAIN data only, then transform both train and test
    This is crucial for LSTM models
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame, optional
        Testing data
    columns : list, optional
        Columns to scale (if None, scales all numeric columns)
    scaler_type : str
        Type of scaler: 'minmax' or 'standard'
        
    Returns:
    --------
    train_scaled, test_scaled, scaler : tuple
        Scaled dataframes and fitted scaler object
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Select columns to scale
    if columns is None:
        columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit on train data only
    train_scaled = train_data.copy()
    train_scaled[columns] = scaler.fit_transform(train_data[columns])
    
    # Transform test data if provided
    test_scaled = None
    if test_data is not None:
        test_scaled = test_data.copy()
        test_scaled[columns] = scaler.transform(test_data[columns])
    
    print(f"Data scaling ({scaler_type}):")
    print(f"  Columns scaled: {len(columns)}")
    print(f"  Scaler fitted on train data only")
    if test_data is not None:
        print(f"  Both train and test data transformed")
    
    return train_scaled, test_scaled, scaler


def prepare_for_arima(data, value_col='WPU101704', apply_diff=True):
    """
    Prepare data specifically for ARIMA modeling
    
    Based on Step 2: Series is non-stationary, d=1 recommended
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    value_col : str
        Target column
    apply_diff : bool
        Whether to apply first difference (recommended: True)
        
    Returns:
    --------
    pd.Series
        Prepared series for ARIMA (differenced if apply_diff=True)
    """
    series = data[value_col].copy()
    
    if apply_diff:
        series_diff = series.diff().dropna()
        print(f"ARIMA Preparation:")
        print(f"  - Applied first difference (d=1)")
        print(f"  - Original length: {len(series)}")
        print(f"  - Differenced length: {len(series_diff)}")
        return series_diff
    else:
        print(f"ARIMA Preparation:")
        print(f"  - No differencing applied")
        print(f"  - Length: {len(series)}")
        return series


def prepare_for_prophet(data, date_col='observation_date', value_col='WPU101704'):
    """
    Prepare data for Prophet model
    
    Prophet requires specific column names: 'ds' (date) and 'y' (target)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    date_col : str
        Date column name
    value_col : str
        Target column name
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'ds' and 'y' columns for Prophet
    """
    prophet_df = pd.DataFrame({
        'ds': data[date_col],
        'y': data[value_col]
    })
    
    print(f"Prophet Preparation:")
    print(f"  - Renamed '{date_col}' -> 'ds'")
    print(f"  - Renamed '{value_col}' -> 'y'")
    print(f"  - Shape: {prophet_df.shape}")
    
    return prophet_df


def create_lstm_sequences(data, value_col='WPU101704', sequence_length=12, 
                         target_col=None):
    """
    Create sequences for LSTM model
    
    Parameters:
    -----------
    data : pd.DataFrame or np.array
        Time series data
    value_col : str or None
        Column to use (if data is DataFrame)
    sequence_length : int
        Number of time steps in each sequence (default: 12 months)
    target_col : str, optional
        Target column (if different from value_col)
        
    Returns:
    --------
    X, y : np.array
        Input sequences and target values
        X shape: (samples, sequence_length, features)
        y shape: (samples,)
    """
    # Extract values
    if isinstance(data, pd.DataFrame):
        if target_col is None:
            target_col = value_col
        values = data[value_col].values
        targets = data[target_col].values
    else:
        values = data
        targets = data
    
    X, y = [], []
    
    for i in range(sequence_length, len(values)):
        X.append(values[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"LSTM Sequence Creation:")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Number of sequences: {len(X)}")
    print(f"  - X shape: {X.shape} (samples, timesteps, features)")
    print(f"  - y shape: {y.shape}")
    
    return X, y


def get_preprocessing_summary(train_df, test_df, features_added=None):
    """
    Print summary of preprocessing steps
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Testing dataframe
    features_added : list, optional
        List of feature names that were added
    """
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    
    print(f"\nDataset Sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Total: {len(train_df) + len(test_df)} samples")
    
    print(f"\nTrain Features: {len(train_df.columns)}")
    print(f"Test Features: {len(test_df.columns)}")
    
    if features_added:
        print(f"\nFeatures Added: {len(features_added)}")
        for feat in features_added:
            print(f"  - {feat}")
    
    # Check for NaN values
    train_nan = train_df.isnull().sum().sum()
    test_nan = test_df.isnull().sum().sum()
    
    print(f"\nMissing Values:")
    print(f"  Train: {train_nan} NaNs")
    print(f"  Test: {test_nan} NaNs")
    
    if train_nan > 0 or test_nan > 0:
        print(f"  [WARNING] NaN values present - consider handling before modeling")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
    print("\nAvailable functions:")
    print("  - train_test_split_ts()")
    print("  - create_lag_features()")
    print("  - create_rolling_features()")
    print("  - create_difference_features()")
    print("  - create_time_features()")
    print("  - scale_data()")
    print("  - prepare_for_arima()")
    print("  - prepare_for_prophet()")
    print("  - create_lstm_sequences()")
    print("  - get_preprocessing_summary()")
