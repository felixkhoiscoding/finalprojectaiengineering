# Pre-Computed Forecasts - README

## Overview
This directory contains pre-computed forecasts for fast deployment on Streamlit Cloud.

**Generated:** 2025-11-26 23:12:37  
**Data Period:** June 1982 - September 2025 (520 observations)

## Files Generated

###Forecasts:
- `arima_forecasts.csv` - 666 forecasts (1-36 months)
- `naive_forecasts.csv` - 666 forecasts (1-36 months)
- `prophet_forecasts.csv` - 666 forecasts (1-36 months)
- `lstm_forecasts.csv` - 78 forecasts (1-12 months only)

### Fitted Values (Historical):
- `arima_fitted.csv` - Model fit to historical data
- `naive_fitted.csv` - Naive persistence fit
- `prophet_fitted.csv` - Prophet fit
- `lstm_fitted.csv` - LSTM fit

### Metadata:
- `forecast_metadata.json` - Generation details, model parameters

## File Format

### Forecast Files
```csv
horizon_months,step,date,forecast
1,1,2025-10,268.12
2,1,2025-10,268.23
2,2,2025-11,267.89
...
```

### Fitted Value Files
```csv
date,fitted
1982-06,NaN
1982-07,101.23
...
```

## Why Pre-Computed?

**Benefits:**
1. **Instant Loading:** <1 second vs 30-90 seconds real-time
2. **Cloud-Friendly:** No GPU required on Streamlit Cloud
3. **Consistent Results:** Same forecasts every time
4. **Professional:** Standard practice for production forecasting apps

**Trade-off:**
- Forecasts are static (based on data through Sep 2025)
- To update: Re-run `generate_forecasts.py` with new data

## Usage in Streamlit

The app loads these CSVs instead of training models on-demand:

```python
# Load pre-computed forecasts
arima_df = pd.read_csv('data/forecasts/arima_forecasts.csv')
forecast = arima_df[arima_df['horizon_months'] == user_selection]
```

## Regenerating Forecasts

To regenerate with updated data:

```bash
python generate_forecasts.py
```

**Requirements:**
- Local GPU recommended for LSTM (2-3 minutes)
- Prophet may take 5-10 minutes
- Total runtime: ~5-10 minutes

## Model Details

| Model | Horizons | Count | Notes |
|-------|----------|-------|-------|
| ARIMA | 1-36 months | 666 | Statistical, any horizon |
| Naive | 1-36 months | 666 | Baseline persistence |
| Prophet | 1-36 months | 666 | Seasonality detection |
| LSTM | 1-12 months | 78 | Deep learning, limited horizon |

**LSTM Limitation:** Maximum 12 months due to error accumulation in iterative forecasting.

---
**For thesis documentation:** These forecasts were pre-generated using local GPU resources to enable optimal user experience and cloud deployment feasibility.
