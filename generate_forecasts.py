"""
Pre-compute all forecasts for Streamlit deployment
This script generates forecasts for all models and horizons to enable fast, GPU-free deployment.

Run this locally with GPU before deploying to Streamlit Cloud.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from config.config import RAW_DATA_PATH

print("="*60)
print("FORECAST GENERATION SCRIPT")
print("="*60)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create forecasts directory
forecasts_dir = Path("data/forecasts")
forecasts_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("[DATA] Loading data...")
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')
print(f"   Loaded {len(df)} observations from {df['observation_date'].min()} to {df['observation_date'].max()}")
print()

# Metadata
metadata = {
    "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "data_end_date": df['observation_date'].max().strftime('%Y-%m-%d'),
    "total_observations": len(df),
    "models": {}
}

# ============================================================================
# 1. ARIMA FORECASTS (1-36 months)
# ============================================================================
print("[ARIMA] Generating ARIMA forecasts (1-36 months)...")
from statsmodels.tsa.statespace.sarimax import SARIMAX

arima_forecasts = []
arima_model = SARIMAX(df['WPU101704'], order=(1,1,1))
arima_fitted_model = arima_model.fit(disp=False)

# Get fitted values
arima_fitted = arima_fitted_model.fittedvalues

for months in range(1, 37):
    forecast_vals = arima_fitted_model.forecast(steps=months).values
    last_date = df['observation_date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    
    for i, (date, value) in enumerate(zip(future_dates, forecast_vals)):
        arima_forecasts.append({
            'horizon_months': months,
            'step': i + 1,
            'date': date.strftime('%Y-%m'),
            'forecast': value
        })
    
    if months % 6 == 0:
        print(f"   [OK] Generated {months}-month forecasts")

arima_df = pd.DataFrame(arima_forecasts)
arima_df.to_csv(forecasts_dir / 'arima_forecasts.csv', index=False)

# Save fitted values
arima_fitted_df = pd.DataFrame({
    'date': df['observation_date'].dt.strftime('%Y-%m'),
    'fitted': arima_fitted
})
arima_fitted_df.to_csv(forecasts_dir / 'arima_fitted.csv', index=False)

metadata['models']['ARIMA'] = {
    'horizons': '1-36 months',
    'total_forecasts': len(arima_df),
    'order': '(1,1,1)'
}
print(f"   [DONE] ARIMA complete: {len(arima_df)} forecasts saved")
print()

# ============================================================================
# 2. NAIVE FORECASTS (1-36 months)
# ============================================================================
print("[NAIVE] Generating Naive forecasts (1-36 months)...")

naive_forecasts = []
last_value = df['WPU101704'].iloc[-1]

for months in range(1, 37):
    last_date = df['observation_date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    
    for i, date in enumerate(future_dates):
        naive_forecasts.append({
            'horizon_months': months,
            'step': i + 1,
            'date': date.strftime('%Y-%m'),
            'forecast': last_value
        })

naive_df = pd.DataFrame(naive_forecasts)
naive_df.to_csv(forecasts_dir / 'naive_forecasts.csv', index=False)

# Fitted values (shifted by 1)
naive_fitted_df = pd.DataFrame({
    'date': df['observation_date'].dt.strftime('%Y-%m'),
    'fitted': df['WPU101704'].shift(1)
})
naive_fitted_df.to_csv(forecasts_dir / 'naive_fitted.csv', index=False)

metadata['models']['Naive'] = {
    'horizons': '1-36 months',
    'total_forecasts': len(naive_df),
    'method': 'last_value_persistence'
}
print(f"   [DONE] Naive complete: {len(naive_df)} forecasts saved")
print()

# ============================================================================
# 3. PROPHET FORECASTS (1-36 months)
# ============================================================================
print("[PROPHET] Generating Prophet forecasts (1-36 months)...")
from prophet import Prophet

prophet_forecasts = []

prophet_data = pd.DataFrame({
    'ds': df['observation_date'],
    'y': df['WPU101704']
})

prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(prophet_data)

# Get fitted values
prophet_fitted_result = prophet_model.predict(prophet_data)
prophet_fitted = prophet_fitted_result['yhat'].values

for months in range(1, 37):
    last_date = df['observation_date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    future = pd.DataFrame({'ds': future_dates})
    forecast = prophet_model.predict(future)
    
    for i, (date, value) in enumerate(zip(future_dates, forecast['yhat'].values)):
        prophet_forecasts.append({
            'horizon_months': months,
            'step': i + 1,
            'date': date.strftime('%Y-%m'),
            'forecast': value
        })
    
    if months % 6 == 0:
        print(f"   [OK] Generated {months}-month forecasts")

prophet_df = pd.DataFrame(prophet_forecasts)
prophet_df.to_csv(forecasts_dir / 'prophet_forecasts.csv', index=False)

# Save fitted values
prophet_fitted_df = pd.DataFrame({
    'date': df['observation_date'].dt.strftime('%Y-%m'),
    'fitted': prophet_fitted
})
prophet_fitted_df.to_csv(forecasts_dir / 'prophet_fitted.csv', index=False)

metadata['models']['Prophet'] = {
    'horizons': '1-36 months',
    'total_forecasts': len(prophet_df),
    'parameters': 'yearly_seasonality=True'
}
print(f"   [DONE] Prophet complete: {len(prophet_df)} forecasts saved")
print()

# ============================================================================
# 4. LSTM FORECASTS (1-12 months only)
# ============================================================================
print("[LSTM] Generating LSTM forecasts (1-12 months)...")
print("   [WARNING] This will take 2-3 minutes with GPU training...")
import joblib
from src.advanced_models import train_lstm

lstm_forecasts = []

try:
    # Load LSTM components
    processed_dir = Path("data/processed")
    X_train = np.load(processed_dir / 'X_train_lstm.npy')
    y_train = np.load(processed_dir / 'y_train_lstm.npy')
    X_test = np.load(processed_dir / 'X_test_lstm.npy')
    scaler = joblib.load(processed_dir / 'lstm_scaler.pkl')
    
    # SET RANDOM SEEDS for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    print("   [TRAINING] Training LSTM model...")
    _, lstm_model, _ = train_lstm(X_train, y_train, X_test[:1], epochs=100, batch_size=16)
    
    # Get fitted values for entire dataset
    sequence_length = 12
    all_data = df['WPU101704'].values
    all_data_scaled = scaler.transform(all_data.reshape(-1, 1))
    
    lstm_fitted_list = []
    for i in range(sequence_length, len(all_data)):
        X_input = all_data_scaled[i-sequence_length:i].reshape(1, sequence_length, 1)
        pred_scaled = lstm_model.predict(X_input, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
        lstm_fitted_list.append(pred_value)
    
    lstm_fitted_full = np.full(len(df), np.nan)
    lstm_fitted_full[sequence_length:] = lstm_fitted_list
    
    # Save fitted values
    lstm_fitted_df = pd.DataFrame({
        'date': df['observation_date'].dt.strftime('%Y-%m'),
        'fitted': lstm_fitted_full
    })
    lstm_fitted_df.to_csv(forecasts_dir / 'lstm_fitted.csv', index=False)
    
    # Generate forecasts for 1-12 months
    for months in range(1, 13):
        last_12 = df.tail(12)['WPU101704'].values
        last_12_scaled = scaler.transform(last_12.reshape(-1, 1))
        
        forecast_values = []
        current_sequence = last_12_scaled.copy()
        
        for _ in range(months):
            X_input = current_sequence[-12:].reshape(1, 12, 1)
            pred_scaled = lstm_model.predict(X_input, verbose=0)
            pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
            forecast_values.append(pred_value)
            current_sequence = np.append(current_sequence, pred_scaled)
        
        last_date = df['observation_date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
        
        for i, (date, value) in enumerate(zip(future_dates, forecast_values)):
            lstm_forecasts.append({
                'horizon_months': months,
                'step': i + 1,
                'date': date.strftime('%Y-%m'),
                'forecast': value
            })
        
        if months % 3 == 0:
            print(f"   [OK] Generated {months}-month forecasts")
    
    lstm_df = pd.DataFrame(lstm_forecasts)
    lstm_df.to_csv(forecasts_dir / 'lstm_forecasts.csv', index=False)
    
    metadata['models']['LSTM'] = {
        'horizons': '1-12 months',
        'total_forecasts': len(lstm_df),
        'architecture': 'Sequential LSTM',
        'epochs': 100,
        'limitation': 'Maximum 12 months due to error accumulation'
    }
    print(f"   [DONE] LSTM complete: {len(lstm_df)} forecasts saved")
    
except Exception as e:
    print(f"   [ERROR] LSTM failed: {e}")
    print(f"   [WARNING] LSTM forecasts not generated - will fall back to real-time in app")
    metadata['models']['LSTM'] = {
        'status': 'failed',
        'error': str(e)
    }

print()

# ============================================================================
# Save metadata
# ============================================================================
with open(forecasts_dir / 'forecast_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("="*60)
print("[SUCCESS] ALL FORECASTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nSaved to: {forecasts_dir.absolute()}")
print(f"\nSummary:")
print(f"   - ARIMA:   {metadata['models']['ARIMA']['total_forecasts']} forecasts (1-36 months)")
print(f"   - Naive:   {metadata['models']['Naive']['total_forecasts']} forecasts (1-36 months)")
print(f"   - Prophet: {metadata['models']['Prophet']['total_forecasts']} forecasts (1-36 months)")
if 'total_forecasts' in metadata['models'].get('LSTM', {}):
    print(f"   - LSTM:    {metadata['models']['LSTM']['total_forecasts']} forecasts (1-12 months)")
else:
    print(f"   - LSTM:    Failed (see above)")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n[INFO] Next step: Deploy Streamlit app - forecasts will load instantly!")
print("="*60)
