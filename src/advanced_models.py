"""
Advanced Models Module

ARIMA, Prophet, and LSTM implementations for time-series forecasting.
Target: Beat baseline (RMSE < 8.93, MAPE < 2.89%)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def train_arima(train, test, value_col='WPU101704', order=(1,1,1), seasonal=False):
    """
    Train ARIMA or SARIMA model
    
    FIXED: Uses manual ARIMA instead of auto_arima to avoid parameter errors
    Based on Step 2: d=1 (non-stationary), weak seasonality
    
    Parameters:
    -----------
    train : pd.DataFrame or pd.Series
        Training data
    test : pd.DataFrame or pd.Series 
        Testing data
    value_col : str
        Column name (if DataFrame)
    order : tuple
        ARIMA order (p,d,q) - default (1,1,1)
    seasonal : bool
        Whether to use SARIMA (seasonal ARIMA)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Extract series
    if isinstance(train, pd.DataFrame):
        y_train = train[value_col]
        y_test = test[value_col]
    else:
        y_train = train
        y_test = test
    
    print(f"\nTraining {'SARIMA' if seasonal else 'ARIMA'} model...")
    print(f"Order: {order}")
    
    # Build ARIMA/SARIMA model
    if seasonal:
        # SARIMA with seasonal component (1,1,1,12)
        seasonal_order = (1, 1, 1, 12)
        print(f"Seasonal order: {seasonal_order}")
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        # Simple ARIMA
        model = SARIMAX(
            y_train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    
    # Fit model
    fitted_model = model.fit(disp=False)
    
    print(f"Model fitted successfully")
    print(f"AIC: {fitted_model.aic:.2f}")
    
    # Forecast
    predictions = fitted_model.forecast(steps=len(y_test))
    
    return predictions.values, fitted_model


def train_prophet(train, test, date_col='observation_date', value_col='WPU101704'):
    """
    Train Prophet model
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data with date and value columns
    test : pd.DataFrame
        Testing data
    date_col : str
        Date column name
    value_col : str
        Value column name
    """
    from prophet import Prophet
    
    # Prepare data (ds, y format)
    train_prophet = pd.DataFrame({
        'ds': train[date_col],
        'y': train[value_col]
    })
    
    print("\nTraining Prophet model...")
    
    # Initialize and fit
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train_prophet)
    
    # Create future dataframe
    future = pd.DataFrame({'ds': test[date_col]})
    
    # Predict
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    return predictions, model


def train_lstm(X_train, y_train, X_test, epochs=50, batch_size=16, units=64):
    """
    Train LSTM model
    
    Parameters:
    -----------
    X_train : np.array
        Training sequences (samples, timesteps, features)
    y_train : np.array
        Training targets
    X_test : np.array
        Test sequences
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    units : int
        Number of LSTM units
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    print("\nBuilding LSTM model...")
    
    # Build model
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Model architecture: {units} -> {units//2} -> 1")
    print(f"Training for up to {epochs} epochs...")
    
    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )
    
    print(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    # Predict
    if len(X_test) > 0:
        predictions = model.predict(X_test, verbose=0)
    else:
        predictions = None
    
    return predictions, model, history


def evaluate_advanced_models(train_df, test_df, X_train_lstm=None, y_train_lstm=None, 
                             X_test_lstm=None, y_test_lstm=None, lstm_scaler=None):
    """
    Evaluate all advanced models
    
    Returns dictionary with predictions and metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    results = {}
    y_true = test_df['WPU101704'].values
    
    # 1. ARIMA
    try:
        print("\n" + "="*60)
        print("ARIMA MODEL")
        print("="*60)
        pred_arima, model_arima = train_arima(train_df, test_df, order=(1,1,1), seasonal=False)
        
        mae = mean_absolute_error(y_true, pred_arima)
        rmse = np.sqrt(mean_squared_error(y_true, pred_arima))
        mape = np.mean(np.abs((y_true - pred_arima) / y_true)) * 100
        
        results['ARIMA'] = {
            'predictions': pred_arima,
            'model': model_arima,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        print(f"ARIMA Performance: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    except Exception as e:
        print(f"[ERROR] ARIMA failed: {e}")
        results['ARIMA'] = None
    
    # 2. Prophet
    try:
        print("\n" + "="*60)
        print("PROPHET MODEL")
        print("="*60)
        pred_prophet, model_prophet = train_prophet(train_df, test_df)
        
        mae = mean_absolute_error(y_true, pred_prophet)
        rmse = np.sqrt(mean_squared_error(y_true, pred_prophet))
        mape = np.mean(np.abs((y_true - pred_prophet) / y_true)) * 100
        
        results['Prophet'] = {
            'predictions': pred_prophet,
            'model': model_prophet,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        print(f"Prophet Performance: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    except Exception as e:
        print(f"[ERROR] Prophet failed: {e}")
        results['Prophet'] = None
    
    # 3. LSTM (if data provided)
    if X_train_lstm is not None and len(X_test_lstm) > 0:
        try:
            print("\n" + "="*60)
            print("LSTM MODEL")
            print("="*60)
            pred_lstm_scaled, model_lstm, history = train_lstm(
                X_train_lstm, y_train_lstm, X_test_lstm,
                epochs=50, batch_size=16, units=64
            )
            
            # Inverse transform
            pred_lstm = lstm_scaler.inverse_transform(pred_lstm_scaled).flatten()
            
            mae = mean_absolute_error(y_true, pred_lstm)
            rmse = np.sqrt(mean_squared_error(y_true, pred_lstm))
            mape = np.mean(np.abs((y_true - pred_lstm) / y_true)) * 100
            
            results['LSTM'] = {
                'predictions': pred_lstm,
                'model': model_lstm,
                'history': history,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            print(f"LSTM Performance: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        except Exception as e:
            print(f"[ERROR] LSTM failed: {e}")
            results['LSTM'] = None
    else:
        print("\n[SKIP] LSTM: Insufficient test data for sequences")
        results['LSTM'] = None
    
    return results


if __name__ == "__main__":
    print("Advanced models module loaded!")
    print("Available functions:")
    print("  - train_arima() - FIXED: Manual SARIMAX implementation")
    print("  - train_prophet()")
    print("  - train_lstm()")
    print("  - evaluate_advanced_models()")
